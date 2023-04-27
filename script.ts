// noinspection DuplicatedCode

// @ts-ignore
import type * as ort from 'onnxruntime-common'

declare const ort: typeof import('onnxruntime-common')

const g_expires = 365

function gen_arch_config() {
	const arch_config = {
		swin_unet: undefined,
		cunet: undefined
	}

	{ // swin_unet
		const swin_unet = {}

		for (const domain of ['art', 'photo']) {
			const template = {
				arch: 'swin_unet',
				domain,

				calc_tile_size(tile_size, config) {
					while (true) {
						if ((tile_size - 16) % 12 === 0 && (tile_size - 16) % 16 === 0) {
							break
						}
						tile_size += 1
					}
					return tile_size
				}
			}

			const contents = {
				scale2x: {
					...template,
					scale: 2,
					offset: 16
				},
				scale4x: {
					...template,
					scale: 4,
					offset: 32
				},
				scale1x: {
					...template,
					scale: 1,
					offset: 8
				}
			}

			for (let i = 0; i < 4; i++) {
				contents[`noise${i}_scale2x`] = {
					...template,
					scale: 2,
					offset: 16
				}

				contents[`noise${i}_scale4x`] = {
					...template,
					scale: 4,
					offset: 32
				}

				contents[`noise${i}`] = {
					...template,
					scale: 1,
					offset: 8
				}
			}

			swin_unet[domain] = contents
		}

		arch_config.swin_unet = swin_unet
	}

	{ // cunet
		const template = {
			arch: 'cunet',
			domain: 'art',

			calc_tile_size(tile_size, config) {
				tile_size = tile_size + (config.offset - 16) * 2
				tile_size -= tile_size % 4
				return tile_size
			}
		}

		const art = {
			scale2x: {
				...template,
				scale: 2,
				offset: 36
			},
			scale1x: {
				...template,
				scale: 1,
				offset: 28
			}
		}

		for (let i = 0; i < 4; ++i) {
			art[`noise${i}_scale2x`] = {
				...template,
				scale: 2,
				offset: 36
			}

			art[`noise${i}`] = {
				...template,
				scale: 1,
				offset: 28
			}
		}

		arch_config.cunet = {art}
	}

	return arch_config
}

const CONFIG = {
	arch: gen_arch_config(),

	get_config(arch, style, method) {
		const config = this.arch[arch]?.[style]?.[method]

		if (config?.path === undefined) {
			config.path = `models/${arch}/${style}/${method}.onnx`
		}

		return config
	},

	get_helper_model_path: name => `models/utils/${name}.onnx`
}

const onnx_session = {
	sessions: {},

	async get_session(path): Promise<ort.InferenceSession> {
		return await (this.sessions[path] ?? (this.sessions[path] = ort.InferenceSession.create(path, {
			executionProviders: ['wasm']
		})))
	}
}

const BLEND_SIZE = 16

interface BlendingParameters {
	y_h: number
	y_w: number
	input_offset: number
	input_blend_size: number
	input_tile_step: number
	output_tile_step: number
	h_blocks: number
	w_blocks: number
	y_buffer_h: number
	y_buffer_w: number
	pad: [number, number, number, number]
}

const SeamBlending = class {
	private readonly dims: number
	private readonly scale: number
	private readonly offset: number
	private readonly tile_size: number
	private readonly blend_size: number

	private param: any
	private pixels: ort.TypedTensor<'float32'>
	private weights: ort.TypedTensor<'float32'>
	private blend_filter: ort.TypedTensor<'float32'>
	private output: ort.TypedTensor<'float32'>

	constructor(dims, scale, offset, tile_size, blend_size = BLEND_SIZE) {
		this.dims = dims
		this.scale = scale
		this.offset = offset
		this.tile_size = tile_size
		this.blend_size = blend_size
	}

	async build() {
		this.param = SeamBlending.calc_parameters(this.dims, this.scale, this.offset, this.tile_size, this.blend_size)
		this.pixels = new ort.Tensor('float32', new Float32Array(this.param.y_buffer_h * this.param.y_buffer_w * 3), [3, this.param.y_buffer_h, this.param.y_buffer_w])
		this.weights = new ort.Tensor('float32', new Float32Array(this.param.y_buffer_h * this.param.y_buffer_w * 3), [3, this.param.y_buffer_h, this.param.y_buffer_w])
		this.blend_filter = await this.create_seam_blending_filter()
		this.output = new ort.Tensor('float32', new Float32Array(this.blend_filter.data.length), this.blend_filter.dims)
	}

	update(tensor, tile_y, tile_x) {
		const step_size = this.param.output_tile_step
		const [_, H, W] = this.blend_filter.dims
		const HW = H * W
		const buffer_h = this.pixels.dims[1]
		const buffer_w = this.pixels.dims[2]
		const buffer_hw = buffer_h * buffer_w
		const h_i = step_size * tile_y
		const w_i = step_size * tile_x
		let old_weight,
		    next_weight,
		    new_weight
		for (let channel = 0; channel < 3; ++channel) {
			for (let i = 0; i < H; ++i) {
				for (let j = 0; j < W; ++j) {
					const tile_index = channel * HW + i * W + j
					const buffer_index = channel * buffer_hw + (h_i + i) * buffer_w + (w_i + j)
					old_weight = this.weights.data[buffer_index]
					next_weight = old_weight + this.blend_filter.data[tile_index]
					old_weight /= next_weight
					new_weight = 1.0 - old_weight
					this.pixels.data[buffer_index] = (this.pixels.data[buffer_index] * old_weight + tensor.data[tile_index] * new_weight)
					this.weights.data[buffer_index] += this.blend_filter.data[tile_index]
					this.output.data[tile_index] = this.pixels.data[buffer_index]
				}
			}
		}
		return this.output
	}

	get_rendering_config() {
		return this.param
	}

	static calc_parameters(dims: number, scale: number, offset: number, tile_size: number, blend_size: number): BlendingParameters {
		let p: Partial<BlendingParameters> = {}
		const x_h = dims[2]
		const x_w = dims[3]
		p.y_h = x_h * scale
		p.y_w = x_w * scale
		p.input_offset = Math.ceil(offset / scale)
		p.input_blend_size = Math.ceil(blend_size / scale)
		p.input_tile_step = tile_size - (p.input_offset * 2 + p.input_blend_size)
		p.output_tile_step = p.input_tile_step * scale
		let [h_blocks, w_blocks, input_h, input_w] = [0, 0, 0, 0]
		while (input_h < x_h + p.input_offset * 2) {
			input_h = h_blocks * p.input_tile_step + tile_size
			++h_blocks
		}
		while (input_w < x_w + p.input_offset * 2) {
			input_w = w_blocks * p.input_tile_step + tile_size
			++w_blocks
		}
		p.h_blocks = h_blocks
		p.w_blocks = w_blocks
		p.y_buffer_h = input_h * scale
		p.y_buffer_w = input_w * scale
		p.pad = [p.input_offset, input_w - (x_w + p.input_offset), p.input_offset, input_h - (x_h + p.input_offset)]
		return p as BlendingParameters
	}

	async create_seam_blending_filter(): Promise<ort.TypedTensor<'float32'>> {
		const session = await onnx_session.get_session(CONFIG.get_helper_model_path('create_seam_blending_filter'))
		let scale = new ort.Tensor('int64', BigInt64Array.from([BigInt(this.scale)]), [])
		let offset = new ort.Tensor('int64', BigInt64Array.from([BigInt(this.offset)]), [])
		let tile_size = new ort.Tensor('int64', BigInt64Array.from([BigInt(this.tile_size)]), [])
		let out = await session.run({scale, offset, tile_size})
		return out.y as ort.TypedTensor<'float32'>
	}
}

interface SettingsSnapshot {
	model_name: 'swin_unet.art' | 'swin_unet.photo' | 'cunet',
	noise: -1 | 0 | 1 | 2 | 3,
	scale: 1 | 2 | 4,
	tile_size: number,
	tile_random: boolean,
	tta_level: 0 | 2 | 4,
	detect_alpha: boolean
}

const onnx_runner = {
	stop_flag: false,
	running: false,

	to_input(rgba, width, height, keep_alpha = false) {
		if (keep_alpha) {
			const rgb = new Float32Array(height * width * 3)
			const alpha1 = new Float32Array(height * width)
			const alpha3 = new Float32Array(height * width * 3)
			for (let y = 0; y < height; ++y) {
				for (let x = 0; x < width; ++x) {
					const i = (y * width * 4) + (x * 4)
					const j = (y * width + x)
					rgb[j] = rgba[i] / 255.0
					rgb[j + (height * width)] = rgba[i + 1] / 255.0
					rgb[j + 2 * (height * width)] = rgba[i + 2] / 255.0
					const alpha = rgba[i + 3] / 255.0
					alpha1[j] = alpha
					alpha3[j] = alpha
					alpha3[j + (height * width)] = alpha
					alpha3[j + 2 * (height * width)] = alpha
				}
			}
			return [
				new ort.Tensor('float32', rgb, [1, 3, height, width]),
				new ort.Tensor('float32', alpha1, [1, 1, height, width]),
				new ort.Tensor('float32', alpha3, [1, 3, height, width])
			]
		} else {
			const rgb = new Float32Array(height * width * 3)
			const bg_color = 1.0
			for (let y = 0; y < height; ++y) {
				for (let x = 0; x < width; ++x) {
					const alpha = rgba[(y * width * 4) + (x * 4) + 3] / 255.0
					for (let c = 0; c < 3; ++c) {
						const i = (y * width * 4) + (x * 4) + c
						const j = (y * width + x) + c * (height * width)
						rgb[j] = alpha * (rgba[i] / 255.0) + (1 - alpha) * bg_color
					}
				}
			}
			return [new ort.Tensor('float32', rgb, [1, 3, height, width])]
		}
	},

	to_image_data(z, alpha3, width, height) {
		const rgba = new Uint8ClampedArray(height * width * 4)
		if (alpha3 != null) {
			for (let y = 0; y < height; ++y) {
				for (let x = 0; x < width; ++x) {
					let alpha_v = 0.0
					for (let c = 0; c < 3; ++c) {
						const i = (y * width * 4) + (x * 4) + c
						const j = (y * width + x) + c * (height * width)
						rgba[i] = (z[j] * 255.0) + 0.49999
						alpha_v += alpha3[j] * (1.0 / 3.0)
					}
					rgba[(y * width * 4) + (x * 4) + 3] = (alpha_v * 255.0) + 0.49999
				}
			}
		} else {
			rgba.fill(255)
			for (let y = 0; y < height; ++y) {
				for (let x = 0; x < width; ++x) {
					for (let c = 0; c < 3; ++c) {
						const i = (y * width * 4) + (x * 4) + c
						const j = (y * width + x) + c * (height * width)
						rgba[i] = (z[j] * 255.0) + 0.49999
					}
				}
			}
		}
		return new ImageData(rgba, width, height)
	},

	check_single_color(rgba, keep_alpha = false) {
		const r = rgba[0]
		const g = rgba[1]
		const b = rgba[2]
		const a = rgba[3]
		for (let i = 0; i < rgba.length; i += 4) {
			if (r != rgba[i] || g != rgba[i + 1] || b != rgba[i + 2] || a != rgba[i + 3]) {
				return null
			}
		}
		if (keep_alpha) {
			return [r / 255.0, g / 255.0, b / 255.0, a / 255.0]
		} else {
			const bg_color = 1.0
			const frac = a / 255.0

			return [
				frac * (r / 255.0) + (1 - frac) * bg_color,
				frac * (g / 255.0) + (1 - frac) * bg_color,
				frac * (b / 255.0) + (1 - frac) * bg_color,
				1.0
			]
		}
	},

	check_alpha_channel(rgba) {
		for (let i = 0; i < rgba.length; i += 4) {
			const alpha = rgba[i + 3]
			if (alpha != 255) {
				return true
			}
		}
		return false
	},

	create_single_color_tensor(rgba, size) {
		const rgb = new Float32Array(size * size * 3)
		const alpha3 = new Float32Array(size * size * 3)
		alpha3.fill(rgba[3])
		for (let c = 0; c < 3; ++c) {
			const v = rgba[c]
			for (let i = 0; i < size * size; ++i) {
				rgb[c * size * size + i] = v
			}
		}
		return [new ort.Tensor('float32', rgb, [1, 3, size, size]), new ort.Tensor('float32', alpha3, [1, 3, size, size])]
	},

	shuffleArray(array) {
		for (let i = array.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[array[i], array[j]] = [array[j], array[i]]
		}
	},

	async tiled_render(image_data, model_config, alpha_config, settings: SettingsSnapshot, output_canvas, block_callback) {
		this.stop_flag = false

		if (this.running) {
			console.log('Already running')
			return
		}

		this.running = true

		console.log(`tile size = ${settings.tile_size}`)

		output_canvas.width = image_data.width * model_config.scale
		output_canvas.height = image_data.height * model_config.scale

		const output_ctx = output_canvas.getContext('2d', {willReadFrequently: true})
		const model = await onnx_session.get_session(model_config.path)
		const has_alpha = alpha_config != null
		const alpha_model = has_alpha ? await onnx_session.get_session(alpha_config.path) : null

		const [rgb, alpha1, alpha3] = this.to_input(image_data.data, image_data.width, image_data.height, has_alpha)

		const seam_blending = new SeamBlending(rgb.dims, model_config.scale, model_config.offset, settings.tile_size)
		const seam_blending_alpha = has_alpha ? new SeamBlending(alpha3.dims, model_config.scale, model_config.offset, settings.tile_size) : null

		await Promise.all([seam_blending.build(), seam_blending_alpha?.build()])

		const {pad, h_blocks, w_blocks, input_tile_step, output_tile_step} = seam_blending.get_rendering_config()
		const rgb_padded = await this.padding(
			has_alpha ? await this.alpha_border_padding(rgb, alpha1, BigInt(model_config.offset)) : rgb,
			BigInt(pad[0]), BigInt(pad[1]), BigInt(pad[2]), BigInt(pad[3])
		)

		const alpha_padded = has_alpha ? await this.padding(alpha3, BigInt(pad[0]), BigInt(pad[1]), BigInt(pad[2]), BigInt(pad[3])) : null

		const [, , h, w] = rgb_padded.dims
		image_data = this.to_image_data(rgb_padded.data, alpha_padded?.data, w, h)

		const input_canvas = document.createElement('canvas')
		input_canvas.width = w
		input_canvas.height = h

		const input_ctx = input_canvas.getContext('2d', {
			willReadFrequently: true
		})

		input_ctx.putImageData(image_data, 0, 0)

		const all_blocks = h_blocks * w_blocks
		let progress = 0

		console.time('render')

		const tiles = []
		for (let h_i = 0; h_i < h_blocks; ++h_i) {
			for (let w_i = 0; w_i < w_blocks; ++w_i) {
				const h_in = h_i * input_tile_step
				const w_in = w_i * input_tile_step
				const h_out = h_i * output_tile_step
				const w_out = w_i * output_tile_step
				tiles.push([h_in, w_in, h_out, w_out, h_i, w_i])
			}
		}

		if (settings.tile_random) {
			this.shuffleArray(tiles)
		}

		block_callback(0, all_blocks, true)

		for (let k = 0; k < tiles.length; ++k) {
			let tile: ort.TypedTensor<'float32'>,
			    tile_alpha: ort.TypedTensor<'float32'> | null

			const [h_in, w_in, h_out, w_out, h_i, w_i] = tiles[k]
			const tile_image_data = input_ctx.getImageData(w_in, h_in, settings.tile_size, settings.tile_size)
			const single_color = this.check_single_color(tile_image_data.data, has_alpha)

			if (single_color) {
				[tile, tile_alpha] = this.create_single_color_tensor(single_color, settings.tile_size * model_config.scale - model_config.offset * 2)
			} else {
				[tile, , tile_alpha] = this.to_input(tile_image_data.data, tile_image_data.width, tile_image_data.height, has_alpha)

				if (settings.tta_level > 0) {
					tile = await this.tta_split(tile, BigInt(settings.tta_level))
				}

				tile = (await model.run({x: tile})).y as ort.TypedTensor<'float32'>

				if (settings.tta_level > 0) {
					tile = await this.tta_merge(tile, BigInt(settings.tta_level))
				}

				if (alpha_model) {
					tile_alpha = (await alpha_model.run({x: tile_alpha})).y as ort.TypedTensor<'float32'>
				}
			}

			const rgb = seam_blending.update(tile, h_i, w_i)
			const alpha = has_alpha ? seam_blending_alpha.update(tile_alpha, h_i, w_i) : null
			const output_image_data = this.to_image_data(rgb.data, alpha?.data, tile.dims[3], tile.dims[2])

			output_ctx.putImageData(output_image_data, w_out, h_out)
			progress++

			if (this.stop_flag) {
				block_callback(progress, all_blocks, false)
				this.running = false
				console.timeEnd('render')
				return
			} else {
				block_callback(progress, all_blocks, true)
			}
		}

		console.timeEnd('render')
		this.running = false
	},

	async padding(rgb: ort.TypedTensor<'float32'>, n_left: bigint, n_right: bigint, n_top: bigint, n_bottom: bigint) {
		const pad = await onnx_session.get_session(CONFIG.get_helper_model_path('pad'))
		const left = new ort.Tensor('int64', BigInt64Array.from([n_left]), [])
		const right = new ort.Tensor('int64', BigInt64Array.from([n_right]), [])
		const top = new ort.Tensor('int64', BigInt64Array.from([n_top]), [])
		const bottom = new ort.Tensor('int64', BigInt64Array.from([n_bottom]), [])
		return (await pad.run({x: rgb, left, right, top, bottom})).y as ort.TypedTensor<'float32'>
	},

	async tta_split(rgb: ort.TypedTensor<'float32'>, n_tta_level: bigint) {
		const ses = await onnx_session.get_session(CONFIG.get_helper_model_path('tta_split'))
		const tta_level = new ort.Tensor('int64', BigInt64Array.from([n_tta_level]), [])
		return (await ses.run({x: rgb, tta_level})).y as ort.TypedTensor<'float32'>
	},

	async tta_merge(rgb: ort.TypedTensor<'float32'>, n_tta_level: bigint) {
		const ses = await onnx_session.get_session(CONFIG.get_helper_model_path('tta_merge'))
		const tta_level = new ort.Tensor('int64', BigInt64Array.from([n_tta_level]), [])
		return (await ses.run({x: rgb, tta_level})).y as ort.TypedTensor<'float32'>
	},

	async alpha_border_padding(rgb: ort.TypedTensor<'float32'>, alpha: ort.TypedTensor<'float32'>, n_offset: bigint) {
		const ses = await onnx_session.get_session(CONFIG.get_helper_model_path('alpha_border_padding'))
		rgb = new ort.Tensor('float32', rgb.data, [rgb.dims[1], rgb.dims[2], rgb.dims[3]])
		alpha = new ort.Tensor('float32', alpha.data, [alpha.dims[1], alpha.dims[2], alpha.dims[3]])
		const offset = new ort.Tensor('int64', BigInt64Array.from([n_offset]), [])
		const out = await ses.run({rgb, alpha, offset})
		const out_rgb = out.y as ort.TypedTensor<'float32'>
		return new ort.Tensor('float32', out_rgb.data, [1, out_rgb.dims[0], out_rgb.dims[1], out_rgb.dims[2]])
	},

	async antialias(rgb: ort.TypedTensor<'float32'>) {
		const ses = await onnx_session.get_session(CONFIG.get_helper_model_path('antialias'))
		return (await ses.run({x: rgb})).y as ort.TypedTensor<'float32'>
	}
}

const currentSettings: SettingsSnapshot = {
	model_name: 'swin_unet.art',
	noise: 0,
	scale: 2,
	tile_size: 64,
	tile_random: false,
	tta_level: 0,
	detect_alpha: false
}

const load_settings = () => {
	for (const [key, value] of Object.entries(currentSettings)) {
		if (localStorage.hasOwnProperty(key)) {
			const persisted = localStorage[key]

			switch (typeof value) {
				case 'string':
					currentSettings[key] = persisted
					break
				case 'number':
					currentSettings[key] = parseInt(persisted, 10)
					break
				case 'boolean':
					currentSettings[key] = persisted === 'true'
					break
			}
		}
	}
}

const save_settings = () => {
	for (const [key, value] of Object.entries(currentSettings)) {
		localStorage[key] = value
	}
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
	onLoaded()
} else {
	document.addEventListener('DOMContentLoaded', onLoaded)
}

function onLoaded() {
	ort.env.wasm.proxy = true

	const src = document.getElementById('src') as HTMLCanvasElement
	let src_name = ''
	const dest = document.getElementById('dest') as HTMLCanvasElement

	const start = document.getElementById('start') as HTMLButtonElement
	const stop = document.getElementById('stop') as HTMLButtonElement
	const feedback = document.getElementById('feedback') as HTMLButtonElement

	src.addEventListener('click', () => {
		if (src.style.height === 'auto') {
			src.style.width = `auto`
			src.style.height = `128px`
		} else {
			src.style.width = `auto`
			src.style.height = `auto`
		}
	})

	dest.addEventListener('click', () => {
		if (dest.style.width === 'auto') {
			dest.style.width = `60%`
			dest.style.height = `auto`
		} else {
			dest.style.width = `auto`
			dest.style.height = `auto`
		}
	})

	function set_input_image(file: File) {
		const feedbackWasDisabled = feedback.disabled
		const reader = new FileReader()

		reader.addEventListener('load', () => {
			const img = new Image()
			img.src = reader.result as string
			img.addEventListener('load', () => {
				src.width = img.naturalWidth
				src.height = img.naturalHeight
				src.getContext('2d', {willReadFrequently: true}).drawImage(img, 0, 0)
				src.style.height = '128px'
				src.style.width = 'auto'
				src_name = file.name

				// dest.width = 128
				// dest.height = 128
				// dest.getContext('2d', {willReadFrequently: true}).clearRect(0, 0, dest.width, dest.height)
				// dest.style.width = 'auto'
				// dest.style.height = 'auto'

				start.disabled = false
				feedback.disabled = feedbackWasDisabled
			})
		})

		start.disabled = true
		feedback.disabled = true
		reader.readAsDataURL(file)
	}

	function clear_input_image() {
		src.width = 128
		src.height = 128
		src.getContext('2d', {willReadFrequently: true}).clearRect(0, 0, src.width, src.height)
		src.style.height = 'auto'
		src.style.width = 'auto'

		// dest.width = 128
		// dest.height = 128
		// dest.getContext('2d', {willReadFrequently: true}).clearRect(0, 0, dest.width, dest.height)
		// dest.style.height = 'auto'
		// dest.style.width = 'auto'
	}

	const filePicker = document.getElementById('file') as HTMLInputElement

	filePicker.addEventListener('change', () => {
		if (onnx_runner.running) {
			console.log('Already running')
			return
		}

		if (filePicker.files?.[0]?.type?.match(/image/)) {
			set_input_image(filePicker.files[0])
			set_message('( ・∀・)b')
		} else {
			clear_input_image()
			set_message('(ﾟ∀ﾟ)', 1)
		}
	})

	document.addEventListener('dragover', e => e.preventDefault())
	document.addEventListener('drop', e => {
		if (!e.dataTransfer?.files?.[0]?.type?.match(/image/)) {
			return e.preventDefault()
		}

		if (onnx_runner.running) {
			console.log('Already running')
			return e.preventDefault()
		}

		e.preventDefault()
		filePicker.files = e.dataTransfer.files
		filePicker.dispatchEvent(new Event('change'))
	})

	document.addEventListener('paste', e => {
		if (e.clipboardData?.files?.[0]?.type?.match(/image/)) {
			if (onnx_runner.running) {
				console.log('Already running')
				return e.preventDefault()
			}

			e.preventDefault()
			filePicker.files = e.clipboardData.files
			filePicker.dispatchEvent(new Event('change'))
		}
	})

	async function process() {
		if (onnx_runner.running) {
			console.log('Already running')
			return
		}

		const settings = currentSettings

		const [arch, style] = settings.model_name.split('.')
		let method

		if (settings.scale == 1) {
			if (settings.noise == -1) {
				set_message('(・A・) No Noise Reduction selected!')
				return
			}
			method = `noise${settings.noise}`
		} else {
			if (settings.noise == -1) {
				method = `scale${settings.scale}x`
			} else {
				method = `noise${settings.noise}_scale${settings.scale}x`
			}
		}

		const config = CONFIG.get_config(arch, style, method)

		if (config == null) {
			set_message('(・A・) Model Not found!')
			return
		}

		const image_data = src.getContext('2d', {willReadFrequently: true}).getImageData(0, 0, src.width, src.height)
		const has_alpha = !settings.detect_alpha ? false : onnx_runner.check_alpha_channel(image_data.data)
		const alpha_config = has_alpha ? CONFIG.get_config(arch, style, /scale\d+x/.exec(method)?.[0] ?? 'scale1x') : null

		if (has_alpha && !alpha_config) {
			set_message('(・A・) Model Not found!')
			return
		}

		set_message('(・∀・)φ ... ', -1)

		dest.style.width = 'auto'
		dest.style.height = 'auto'
		feedback.disabled = true

		const tile_size = config.calc_tile_size(settings.tile_size, config)

		const formatTime = secs => `${Math.floor(secs / 60).toString(10).padStart(2, '0')}:${Math.floor(secs % 60).toString(10).padStart(2, '0')}`
		const start = performance.now()

		await onnx_runner.tiled_render(image_data, config, alpha_config, {...settings, tile_size}, dest, (progress, max_progress, processing) => {
			const now = performance.now()
			const spent = (now - start) / 1000
			const eta = (max_progress - progress) / progress * spent

			if (processing) {
				let progress_message = `(${progress}/${max_progress})`

				if (progress > 0) {
					progress_message += `- ${formatTime(spent)} spent - ${formatTime(eta)} remaining`
				}

				loop_message(['( ・∀・)' + (progress % 2 == 0 ? 'φ　 ' : ' φ　') + progress_message, '( ・∀・)' + (progress % 2 != 0 ? 'φ　 ' : ' φ　') + progress_message], 0.5)
			} else {
				set_message('(・A・)!!', 1)
			}
		})

		if (!onnx_runner.stop_flag) {
			const end = performance.now()
			const total = (end - start) / 1000

			dest.toBlob((blob) => {
				const url = URL.createObjectURL(blob)
				const filename = (src_name.split(/(?=\.[^.]+$)/))[0] + '_waifu2x_' + method + '.png'
				set_message(`( ・∀・)つ　<a href="${url}" download="${filename}">Download</a> - took ${formatTime(total)} (${Math.ceil(total * 1000)}ms)`, -1, true)
				feedback.disabled = false
			}, 'image/png')
		}
	}

	const message = document.getElementById('message')

	function set_message(text, second = 2, html = false) {
		if (html) {
			message.innerHTML = text
		} else {
			message.innerText = text
		}

		if (second > 0) {
			const text_node = document.createTextNode('')
			message.appendChild(text_node)

			setTimeout(() => {
				if (text_node.parentNode == message) {
					message.innerText = '( ・∀・)'
				}
			}, second * 1000)
		}
	}

	function loop_message(texts, second = 0.5) {
		message.innerText = texts[0]

		const text_node = document.createTextNode('')
		message.appendChild(text_node)

		let id, i = 0

		id = setInterval(() => {
			i++
			i %= texts.length

			if (text_node.parentNode !== message) {
				clearInterval(id)
			} else {
				message.innerText = texts[i]
				message.appendChild(text_node)
			}
		}, second * 1000)
	}

	start.addEventListener('click', async () => {
		if (filePicker.value !== '') {
			await process()
		} else {
			set_message('(ﾟ∀ﾟ) No Image Found')
		}
	})

	stop.addEventListener('click', () => {
		onnx_runner.stop_flag = true
	})

	feedback.addEventListener('click', () => {
		feedback.disabled = true
		filePicker.files = null
		src.width = dest.width
		src.height = dest.height
		src.getContext('2d', {willReadFrequently: true}).drawImage(dest, 0, 0)
		src.style.width = 'auto'
		src.style.height = '128px'
	})

	load_settings()
	save_settings()

	document.getElementsByName('model').forEach(elem => {
		if (elem instanceof HTMLSelectElement) {
			elem.value = currentSettings.model_name

			elem.addEventListener('change', () => {
				currentSettings.model_name = elem.value as any
				save_settings()

				const is_swin_unet = currentSettings.model_name.split('.')?.[0] === 'swin_unet'

				document.getElementsByName('scale').forEach(elem => {
					if (elem instanceof HTMLSelectElement) {
						const scale_4x = elem.options.namedItem('scale_4x')
						const scale_2x = elem.options.namedItem('scale_2x')

						if (elem.selectedIndex == scale_4x.index && !is_swin_unet) {
							elem.selectedIndex = scale_2x.index
							elem.dispatchEvent(new Event('change'))
						}

						scale_4x.disabled = !is_swin_unet
					}
				})

				document.getElementById('scale-comment').style.display = is_swin_unet ? 'none' : 'unset'

				document.getElementById('tile-comment').style.display =
					currentSettings.model_name.split('.')?.[1] == 'photo' && currentSettings.tile_size < 256 ? 'unset' : 'none'
			})

			elem.dispatchEvent(new Event('change'))
		}
	})

	document.getElementsByName('noise_level').forEach(elem => {
		if (elem instanceof HTMLSelectElement) {
			elem.value = currentSettings.noise.toString()

			elem.addEventListener('change', () => {
				currentSettings.noise = parseInt(elem.value, 10) as any
				save_settings()
			})
		}
	})

	document.getElementsByName('scale').forEach(elem => {
		if (elem instanceof HTMLSelectElement) {
			elem.value = currentSettings.scale.toString()

			elem.addEventListener('change', () => {
				currentSettings.scale = parseInt(elem.value, 10) as any
				save_settings()
			})
		}
	})

	document.getElementsByName('tile_size').forEach(elem => {
		if (elem instanceof HTMLSelectElement) {
			elem.value = currentSettings.tile_size.toString()

			elem.addEventListener('change', () => {
				currentSettings.tile_size = parseInt(elem.value, 10) as any
				save_settings()

				document.getElementById('tile-comment').style.display =
					currentSettings.model_name.split('.')?.[1] == 'photo' && currentSettings.tile_size < 256 ? 'unset' : 'none'
			})
		}
	})

	document.getElementsByName('tile_random').forEach(elem => {
		if (elem instanceof HTMLInputElement && elem.type === 'checkbox') {
			elem.checked = currentSettings.tile_random

			elem.addEventListener('change', () => {
				currentSettings.tile_random = elem.checked
				save_settings()
			})
		}
	})

	document.getElementsByName('tta').forEach(elem => {
		if (elem instanceof HTMLSelectElement) {
			elem.value = currentSettings.tta_level.toString()

			elem.addEventListener('change', () => {
				currentSettings.tta_level = parseInt(elem.value, 10) as any
				save_settings()
			})
		}
	})

	document.getElementsByName('alpha').forEach(elem => {
		if (elem instanceof HTMLSelectElement) {
			elem.value = currentSettings.detect_alpha ? '1' : '0'

			elem.addEventListener('change', () => {
				currentSettings.detect_alpha = elem.value === '1'
				save_settings()
			})
		}
	})

	window.addEventListener('unhandledrejection', function(e) {
		set_message('(-_-) Error: ' + e.reason, -1)
		onnx_runner.running = false
		onnx_runner.stop_flag = false
	})
}
