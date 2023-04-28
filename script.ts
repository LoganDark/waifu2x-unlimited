interface ModelConfig {
	scale: bigint
	offset: bigint
	path: string

	round_tile_size(this: ModelConfig, tile_size: bigint): bigint
}

interface ModelMethods {
	[method: string]: Omit<ModelConfig, 'path'> & Partial<Pick<ModelConfig, 'path'>>
}

interface ModelStyles {
	[style: string]: ModelMethods
}

interface Models {
	[arch: string]: ModelStyles
}

const Models = {
	models: (() => {
		const models: Models = {}

		{ // swin_unet
			const swin_unet: ModelStyles = {}

			for (const style of ['art', 'photo'] as const) {
				const template: Omit<ModelConfig, 'scale' | 'offset' | 'path'> = {
					round_tile_size(tile_size) {
						// closest common denominator of 12 and 16
						const mul = (tile_size + 15n) / 16n

						for (let i = 0n; ; i++) {
							const proposed = (mul + i) * 16n
							if ((proposed - 16n) % 12n == 0n) return proposed
						}
					}
				}

				const methods: ModelMethods = {
					scale2x: {
						...template,
						scale: 2n,
						offset: 16n
					},
					scale4x: {
						...template,
						scale: 4n,
						offset: 32n
					},
					scale1x: {
						...template,
						scale: 1n,
						offset: 8n
					}
				}

				for (let i = 0; i < 4; i++) {
					methods[`noise${i}_scale2x`] = {
						...template,
						scale: 2n,
						offset: 16n
					}

					methods[`noise${i}_scale4x`] = {
						...template,
						scale: 4n,
						offset: 32n
					}

					methods[`noise${i}`] = {
						...template,
						scale: 1n,
						offset: 8n
					}
				}

				swin_unet[style] = methods
			}

			models.swin_unet = swin_unet
		}

		{ // cunet
			const template: Omit<ModelConfig, 'scale' | 'offset' | 'path'> = {
				round_tile_size(tile_size) {
					tile_size = tile_size + (this.offset - 16n) * 2n
					tile_size -= tile_size % 4n
					return tile_size
				}
			}

			const art: ModelMethods = {
				scale2x: {
					...template,
					scale: 2n,
					offset: 36n
				},
				scale1x: {
					...template,
					scale: 1n,
					offset: 28n
				}
			}

			for (let i = 0; i < 4; i++) {
				art[`noise${i}_scale2x`] = {
					...template,
					scale: 2n,
					offset: 36n
				}

				art[`noise${i}`] = {
					...template,
					scale: 1n,
					offset: 28n
				}
			}

			models.cunet = {art}
		}

		return models
	})(),

	get_parameters(arch: string, style: string, method: string): ModelConfig | null {
		const config = this.models[arch]?.[style]?.[method] ?? null

		if (config !== null) {
			config.path = `models/${arch}/${style}/${method}.onnx`

			return config as ModelConfig
		}

		return null
	},

	get_utility_path: (name: string) => `models/utils/${name}.onnx`
}

const BLEND_SIZE = 16n

interface Padding {
	left: bigint
	right: bigint
	top: bigint
	bottom: bigint
}

interface TileParameters {
	col: bigint
	row: bigint

	in_x: bigint
	in_y: bigint
	in_w: bigint
	in_h: bigint

	out_x: bigint
	out_y: bigint
	out_w: bigint
	out_h: bigint
}

class TilingParameters {
	public readonly scale: bigint
	public readonly offset: bigint
	public readonly tile_size: bigint
	public readonly blend_size: bigint
	public readonly output_tile_height: bigint
	public readonly output_tile_width: bigint
	public readonly input_offset: bigint
	public readonly input_blend_size: bigint
	public readonly input_tile_step: bigint
	public readonly output_tile_step: bigint
	public readonly num_h_blocks: bigint
	public readonly num_w_blocks: bigint
	public readonly output_buffer_h: number
	public readonly output_buffer_w: number
	public readonly padding: Padding

	public constructor(input_width: bigint, input_height: bigint, scale: bigint, offset: bigint, tile_size: bigint, blend_size: bigint = BLEND_SIZE) {
		this.scale = scale
		this.offset = offset
		this.tile_size = tile_size
		this.blend_size = blend_size

		this.output_tile_height = input_height * scale
		this.output_tile_width = input_width * scale
		this.input_offset = (offset + scale - 1n) / scale
		this.input_blend_size = (blend_size + scale - 1n) / scale
		this.input_tile_step = tile_size - (this.input_offset * 2n + this.input_blend_size)
		this.output_tile_step = this.input_tile_step * scale

		let [h_blocks, w_blocks, input_h, input_w] = [0n, 0n, 0n, 0n]

		while (input_h < input_height + this.input_offset * 2n) {
			input_h = h_blocks * this.input_tile_step + tile_size
			++h_blocks
		}

		while (input_w < input_width + this.input_offset * 2n) {
			input_w = w_blocks * this.input_tile_step + tile_size
			++w_blocks
		}

		this.num_h_blocks = h_blocks
		this.num_w_blocks = w_blocks
		this.output_buffer_h = Number(input_h * scale)
		this.output_buffer_w = Number(input_w * scale)

		this.padding = {
			left: BigInt(this.input_offset),
			right: BigInt(input_w - (input_width + this.input_offset)),
			top: BigInt(this.input_offset),
			bottom: BigInt(input_h - (input_height + this.input_offset))
		}
	}

// for (let h_i = 0n; h_i < num_h_blocks; h_i++) {
// 	for (let w_i = 0n; w_i < num_w_blocks; w_i++) {
// 		const h_in = h_i * input_tile_step
// 		const w_in = w_i * input_tile_step
// 		const h_out = h_i * output_tile_step
// 		const w_out = w_i * output_tile_step
// 		tiles.push([h_in, w_in, h_out, w_out, h_i, w_i] as const)
// 	}
// }

	public get_tile_params(row: bigint, col: bigint): TileParameters {
		return {
			col,
			row,

			in_x: col * this.input_tile_step,
			in_y: row * this.input_tile_step,
			in_w: this.tile_size,
			in_h: this.tile_size,

			out_x: col * this.output_tile_step,
			out_y: row * this.output_tile_step,
			out_w: this.tile_size * this.scale,
			out_h: this.tile_size * this.scale
		}
	}
}

class SeamBlender {
	private constructor(
		private readonly params: TilingParameters,
		private readonly image_pixels: ort.TypedTensor<'float32'>,
		private readonly image_weights: ort.TypedTensor<'float32'>,
		private readonly tile_filter: ort.TypedTensor<'float32'>,
		private readonly tile_pixels: ort.TypedTensor<'float32'>
	) {}

	public static async create(params: TilingParameters) {
		const {output_buffer_h, output_buffer_w, scale, offset, tile_size} = params
		const tile_filter_promise = SeamBlender.create_tile_filter(scale, offset, tile_size)
		const image_pixels = new ort.Tensor('float32', new Float32Array(output_buffer_h * output_buffer_w * 3), [3, output_buffer_h, output_buffer_w])
		const image_weights = new ort.Tensor('float32', new Float32Array(output_buffer_h * output_buffer_w * 3), [3, output_buffer_h, output_buffer_w])
		const tile_filter = await tile_filter_promise
		const tile_pixels = new ort.Tensor('float32', new Float32Array(tile_filter.data.length), tile_filter.dims)
		return new SeamBlender(params, image_pixels, image_weights, tile_filter, tile_pixels)
	}

	private static async create_tile_filter(scale: bigint, offset: bigint, tile_size: bigint) {
		const model_promise = OnnxSessionsCache.get_or_create_helper('create_seam_blending_filter')
		const scale_tensor = new ort.Tensor('int64', BigInt64Array.from([scale]), [])
		const offset_tensor = new ort.Tensor('int64', BigInt64Array.from([offset]), [])
		const tile_size_tensor = new ort.Tensor('int64', BigInt64Array.from([tile_size]), [])
		const result = await (await model_promise).run({
			scale: scale_tensor,
			offset: offset_tensor,
			tile_size: tile_size_tensor
		})

		return result.y as ort.TypedTensor<'float32'>
	}

	public blend(tile: ort.TypedTensor<'float32'>, tile_x: bigint, tile_y: bigint) {
		const [, tile_h, tile_w] = this.tile_pixels!.dims
		const tile_size = tile_w * tile_h

		const [, image_h, image_w] = this.image_pixels!.dims
		const image_size = image_w * image_h

		const step_size = this.params!.output_tile_step
		const [output_tile_x, output_tile_y] = [step_size * tile_x, step_size * tile_y].map(Number)

		const tile_data = tile.data
		const tile_filter_data = this.tile_filter!.data
		const tile_pixels_data = this.tile_pixels!.data
		const image_weights_data = this.image_weights!.data
		const image_pixels_data = this.image_pixels!.data

		for (let channel = 0; channel < 3; channel++) {
			const tile_offset = channel * tile_size
			const image_offset = channel * image_size

			const tile_channel = tile_data.subarray(tile_offset)
			const tile_filter_channel = tile_filter_data.subarray(tile_offset)
			const tile_pixels_channel = tile_pixels_data.subarray(tile_offset)
			const image_weights_channel = image_weights_data.subarray(image_offset)
			const image_pixels_channel = image_pixels_data.subarray(image_offset)

			for (let filter_y = 0; filter_y < tile_h; filter_y++) {
				const tile_offset = filter_y * tile_w
				const image_offset = (output_tile_y + filter_y) * image_w + output_tile_x

				for (let filter_x = 0; filter_x < tile_w; filter_x++) {
					const tile_index = tile_offset + filter_x
					const image_index = image_offset + filter_x

					const old_factor = image_weights_channel[image_index] / (image_weights_channel[image_index] += tile_filter_channel[tile_index])
					tile_pixels_channel[tile_index] = image_pixels_channel[image_index] = (image_pixels_channel[image_index] * old_factor + tile_channel[tile_index] * (1 - old_factor))
				}
			}
		}

		return this.tile_pixels
	}
}

interface SettingsSnapshot {
	model_name: 'swin_unet.art' | 'swin_unet.photo' | 'cunet'
	noise: -1n | 0n | 1n | 2n | 3n
	scale: 1n | 2n | 4n
	tile_size: bigint
	tile_random: boolean
	tta_level: 0n | 2n | 4n
	detect_alpha: boolean
}

const OnnxSessionsCache = {
	sessions: {} as {[path: string]: ReturnType<typeof ort.InferenceSession.create>},

	async get_or_create(path: string) {
		return await (this.sessions[path] ?? (this.sessions[path] = ort.InferenceSession.create(path, {
			executionProviders: ['wasm']
		})))
	},

	get_or_create_helper(name: string) {
		return this.get_or_create(Models.get_utility_path(name))
	}
}

const onnx_runner = {
	stop_flag: false,
	running: false,

	to_input(image_data: Uint8ClampedArray, width: number, height: number, keep_alpha = false) {
		const len = width * height
		const rgb = new Float32Array(len * 3)
		const alpha1 = new Float32Array(len)

		for (let offset = 0; offset < len; offset += width) {
			for (let x = 0; x < width; x++) {
				const i = offset + x
				const i4 = i * 4
				rgb[i] = image_data[i4] / 255
				rgb[i + len] = image_data[i4 + 1] / 255
				rgb[i + len * 2] = image_data[i4 + 2] / 255
				alpha1[i] = image_data[i4 + 3] / 255
			}
		}

		if (keep_alpha) {
			const alpha3 = new Float32Array(len * 3)
			alpha3.set(alpha1)
			alpha3.set(alpha1, len)
			alpha3.set(alpha1, len * 2)

			return [
				new ort.Tensor('float32', rgb, [1, 3, height, width]),
				new ort.Tensor('float32', alpha1, [1, 1, height, width]),
				new ort.Tensor('float32', alpha3, [1, 3, height, width])
			] as const
		} else {
			return [
				new ort.Tensor('float32', rgb, [1, 3, height, width]),
				null,
				null
			] as const
		}
	},

	to_image_data(rgb_data: Float32Array, alpha3_data: Float32Array | null, width: number, height: number) {
		const len = width * height
		const image_data = new Uint8ClampedArray(len * 4)

		for (let y = 0; y < height; y++) {
			const offset = y * width

			for (let x = 0; x < width; x++) {
				const i = offset + x
				const i4 = i * 4

				image_data[i4] = rgb_data[i] * 255
				image_data[i4 + 1] = rgb_data[i + len] * 255
				image_data[i4 + 2] = rgb_data[i + len * 2] * 255
				image_data[i4 + 3] = (alpha3_data ? (alpha3_data[i] + alpha3_data[i + len] + alpha3_data[i + len * 2]) / 3 : 1) * 255
			}
		}

		return new ImageData(image_data, width, height)
	},

	is_solid_color(pixels: Uint8ClampedArray, keep_alpha = false) {
		const a = pixels[3] / 255,
		      r = pixels[0] * a,
		      g = pixels[1] * a,
		      b = pixels[2] * a

		const len = pixels.length
		for (let i = 4; i < len; i += 4) {
			const a2 = pixels[i + 3] / 255,
			      r2 = pixels[i] * a2,
			      g2 = pixels[i + 1] * a2,
			      b2 = pixels[i + 2] * a2

			if (r2 !== r || g2 !== g || b2 !== b || (keep_alpha && a2 !== a)) return null
		}

		if (keep_alpha) {
			return [r / 255, g / 255, b / 255, a] as const
		} else {
			return [r / 255 + (1 - a), g / 255 + (1 - a), b / 255 + (1 - a), 1] as const
		}
	},

	has_transparency(pixels: Uint8ClampedArray) {
		for (let i = 3; i < pixels.length; i += 4) {
			if (pixels[i] !== 255) return true
		}

		return false
	},

	create_solid_color_tensor(color: readonly [number, number, number, number], tile_size: number) {
		const len = tile_size * tile_size

		const rgb = new Float32Array(len * 3)
		rgb.fill(color[0], 0, len)
		rgb.fill(color[1], len, len * 2)
		rgb.fill(color[2], len * 2, len * 3)

		const alpha3 = new Float32Array(len * 3)
		alpha3.fill(color[3])

		return [
			new ort.Tensor('float32', rgb, [1, 3, tile_size, tile_size]),
			new ort.Tensor('float32', alpha3, [1, 3, tile_size, tile_size])
		] as const
	},

	async tiled_render(
		image_data: ImageData,
		model_config: ModelConfig,
		alpha_config: ModelConfig | null,
		settings: SettingsSnapshot,
		output_canvas: HTMLCanvasElement,
		block_callback: (tile: number, total_tiles: number, processing: boolean) => void
	) {
		this.stop_flag = false

		if (this.running) {
			console.log('Already running')
			return
		}

		this.running = true

		console.log(`tile size = ${settings.tile_size}`)

		const output_ctx = output_canvas.getContext('2d', {willReadFrequently: true})!
		output_canvas.width = Number(BigInt(image_data.width) * model_config.scale)
		output_canvas.height = Number(BigInt(image_data.height) * model_config.scale)

		const model = await OnnxSessionsCache.get_or_create(model_config.path)
		const has_alpha = alpha_config != null
		const alpha_model = has_alpha ? await OnnxSessionsCache.get_or_create(alpha_config.path) : null

		const params = new TilingParameters(BigInt(image_data.width), BigInt(image_data.height), model_config.scale, model_config.offset, settings.tile_size)

		const [blender, alpha_blender] = await Promise.all([SeamBlender.create(params), has_alpha ? SeamBlender.create(params) : null])

		const [rgb, alpha1, alpha3] = this.to_input(image_data.data, image_data.width, image_data.height, has_alpha)
		const {padding} = params

		// noinspection ES6MissingAwait
		const alpha3_padded = alpha3 ? await this.padding(alpha3, padding) : null
		const rgb_padded = await this.padding(alpha1 ? await this.alpha_border_padding(rgb, alpha1, model_config.offset) : rgb, padding)

		const [, , h_padded, w_padded] = rgb_padded.dims
		const input_canvas = document.createElement('canvas')
		const input_ctx = input_canvas.getContext('2d', {willReadFrequently: true})!
		input_canvas.width = w_padded
		input_canvas.height = h_padded
		input_ctx.putImageData(this.to_image_data(rgb_padded.data, alpha3_padded?.data ?? null, w_padded, h_padded), 0, 0)

		const all_blocks = params.num_w_blocks * params.num_h_blocks
		let progress = 0

		console.time('render')

		const tiles = []
		for (let row = 0n; row < params.num_h_blocks; row++) {
			for (let col = 0n; col < params.num_w_blocks; col++) {
				tiles.push([row, col] as const)
			}
		}

		block_callback(0, Number(all_blocks), true)

		while (tiles.length > 0) {
			let tile: ort.TypedTensor<'float32'>,
			    tile_alpha: ort.TypedTensor<'float32'> | null

			const [[row, col]] = tiles.splice(settings.tile_random ? Math.floor(Math.random() * tiles.length) : 0, 1)
			const tile_params = params.get_tile_params(row, col)
			const tile_image_data = input_ctx.getImageData(Number(tile_params.in_x), Number(tile_params.in_y), Number(tile_params.in_w), Number(tile_params.in_h))
			const single_color = this.is_solid_color(tile_image_data.data, has_alpha)

			if (single_color) {
				[tile, tile_alpha] = this.create_solid_color_tensor(single_color, Number(settings.tile_size * model_config.scale - model_config.offset * 2n))
			} else {
				[tile, , tile_alpha] = this.to_input(tile_image_data.data, tile_image_data.width, tile_image_data.height, has_alpha)

				// noinspection ES6MissingAwait
				const tile_alpha_promise = alpha_model && tile_alpha ? alpha_model.run({x: tile_alpha}) : null

				if (settings.tta_level > 0) {
					tile = await this.tta_split(tile, BigInt(settings.tta_level))
				}

				tile = (await model.run({x: tile})).y as ort.TypedTensor<'float32'>

				if (settings.tta_level > 0) {
					tile = await this.tta_merge(tile, BigInt(settings.tta_level))
				}

				tile_alpha = ((await tile_alpha_promise)?.y ?? null) as ort.TypedTensor<'float32'> | null
			}

			const rgb = blender.blend(tile, tile_params.col, tile_params.row)
			const alpha = alpha_blender && tile_alpha ? alpha_blender.blend(tile_alpha, tile_params.col, tile_params.row) : null
			const output_image_data = this.to_image_data(rgb.data, alpha?.data ?? null, tile.dims[3], tile.dims[2])

			output_ctx.putImageData(output_image_data, Number(tile_params.out_x), Number(tile_params.out_y))
			progress++

			if (this.stop_flag) {
				block_callback(progress, Number(all_blocks), false)
				this.running = false
				console.timeEnd('render')
				return
			} else {
				block_callback(progress, Number(all_blocks), true)
			}
		}

		console.timeEnd('render')
		this.running = false
	},

	async padding(rgb: ort.TypedTensor<'float32'>, padding: Padding) {
		const model = await OnnxSessionsCache.get_or_create_helper('pad')
		const [left, right, top, bottom] = [padding.left, padding.right, padding.top, padding.bottom]
			.map(amount => new ort.Tensor('int64', BigInt64Array.from([amount]), []))
		return (await model.run({x: rgb, left, right, top, bottom})).y as ort.TypedTensor<'float32'>
	},

	async tta_split(rgb: ort.TypedTensor<'float32'>, tta_level: bigint) {
		const model = await OnnxSessionsCache.get_or_create_helper('tta_split')

		return (await model.run({
			x: rgb,
			tta_level: new ort.Tensor('int64', BigInt64Array.from([tta_level]), [])
		})).y as ort.TypedTensor<'float32'>
	},

	async tta_merge(rgb: ort.TypedTensor<'float32'>, tta_level: bigint) {
		const model = await OnnxSessionsCache.get_or_create_helper('tta_merge')

		return (await model.run({
			x: rgb,
			tta_level: new ort.Tensor('int64', BigInt64Array.from([tta_level]), [])
		})).y as ort.TypedTensor<'float32'>
	},

	async alpha_border_padding(rgb: ort.TypedTensor<'float32'>, alpha: ort.TypedTensor<'float32'>, offset: bigint) {
		const model = await OnnxSessionsCache.get_or_create_helper('alpha_border_padding')

		const out = await model.run({
			rgb: new ort.Tensor('float32', rgb.data, rgb.dims.slice(1)),
			alpha: new ort.Tensor('float32', alpha.data, alpha.dims.slice(1)),
			offset: new ort.Tensor('int64', BigInt64Array.from([offset]), [])
		})

		const out_rgb = out.y as ort.TypedTensor<'float32'>
		return new ort.Tensor('float32', out_rgb.data, [1, ...out_rgb.dims])
	},

	async antialias(rgb: ort.TypedTensor<'float32'>) {
		const ses = await OnnxSessionsCache.get_or_create_helper('antialias')
		return (await ses.run({x: rgb})).y as ort.TypedTensor<'float32'>
	}
}

const currentSettings: SettingsSnapshot = {
	model_name: 'swin_unet.art',
	noise: 0n,
	scale: 2n,
	tile_size: 64n,
	tile_random: false,
	tta_level: 0n,
	detect_alpha: false
}

const load_settings = () => {
	for (const [key, value] of Object.entries(currentSettings)) {
		if (localStorage.hasOwnProperty(key)) {
			const persisted = localStorage[key]

			switch (typeof value) {
				case 'string':
					currentSettings[key as never] = persisted as never
					break
				case 'number':
					currentSettings[key as never] = parseInt(persisted, 10) as never
					break
				case 'bigint':
					currentSettings[key as never] = BigInt(persisted) as never
					break
				case 'boolean':
					currentSettings[key as never] = (persisted === 'true') as never
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

	let filename = ''

	const src = document.getElementById('src') as HTMLCanvasElement
	const src_ctx = src.getContext('2d', {willReadFrequently: true})!
	const dest = document.getElementById('dest') as HTMLCanvasElement

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

	const start_btn = document.getElementById('start') as HTMLButtonElement
	const stop_btn = document.getElementById('stop') as HTMLButtonElement
	const loop_btn = document.getElementById('loop_btn') as HTMLButtonElement

	function set_input_image(file: File) {
		const feedbackWasDisabled = loop_btn.disabled
		const reader = new FileReader()

		reader.addEventListener('load', () => {
			const img = new Image()
			img.src = reader.result as string
			img.addEventListener('load', () => {
				src.width = img.naturalWidth
				src.height = img.naturalHeight
				src_ctx.drawImage(img, 0, 0)
				src.style.height = '128px'
				src.style.width = 'auto'
				filename = file.name

				start_btn.disabled = false
				loop_btn.disabled = feedbackWasDisabled
			})
		})

		start_btn.disabled = true
		loop_btn.disabled = true
		reader.readAsDataURL(file)
	}

	function clear_input_image() {
		src.width = 128
		src.height = 128
		src_ctx.clearRect(0, 0, src.width, src.height)
		src.style.height = 'auto'
		src.style.width = 'auto'
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

	const message = document.getElementById('message')!

	function set_message(text: string, timeout = 2, html = false) {
		if (html) {
			message.innerHTML = text
		} else {
			message.innerText = text
		}

		if (timeout > 0) {
			const text_node = document.createTextNode('')
			message.appendChild(text_node)

			setTimeout(() => {
				if (text_node.parentNode == message) {
					message.innerText = '( ・∀・)'
				}
			}, timeout * 1000)
		}
	}

	function loop_message(texts: string[], second = 0.5) {
		message.innerText = texts[0]

		const text_node = document.createTextNode('')
		message.appendChild(text_node)

		let id: number, i = 0

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

	async function process() {
		if (onnx_runner.running) {
			console.log('Already running')
			return
		}

		const settings = currentSettings

		const [arch, style] = settings.model_name.split('.')
		let method: string

		if (settings.scale == 1n) {
			if (settings.noise == -1n) {
				set_message('(・A・) No Noise Reduction selected!')
				return
			}

			method = `noise${settings.noise}`
		} else {
			if (settings.noise == -1n) {
				method = `scale${settings.scale}x`
			} else {
				method = `noise${settings.noise}_scale${settings.scale}x`
			}
		}

		const config = Models.get_parameters(arch, style, method)

		if (config == null) {
			set_message('(・A・) Model Not found!')
			return
		}

		const image_data = src_ctx.getImageData(0, 0, src.width, src.height)
		const has_alpha = !settings.detect_alpha ? false : onnx_runner.has_transparency(image_data.data)
		const alpha_config = has_alpha ? Models.get_parameters(arch, style, /scale\d+x/.exec(method)?.[0] ?? 'scale1x') : null

		if (has_alpha && !alpha_config) {
			set_message('(・A・) Alpha Model Not found!')
			return
		}

		set_message('(・∀・)φ ... ', -1)

		dest.style.width = 'auto'
		dest.style.height = 'auto'
		loop_btn.disabled = true

		const tile_size = config.round_tile_size(settings.tile_size)

		const formatTime = (secs: number) => `${Math.floor(secs / 60).toString(10).padStart(2, '0')}:${Math.floor(secs % 60).toString(10).padStart(2, '0')}`
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
				if (blob != null) {
					const url = URL.createObjectURL(blob)
					const download_filename = (filename.split(/(?=\.[^.]+$)/))[0] + '_waifu2x_' + method + '.png'
					set_message(`( ・∀・)つ　<a href="${url}" download="${download_filename}">Download</a> - took ${formatTime(total)} (${Math.ceil(total * 1000)}ms)`, -1, true)
				} else {
					set_message('(・A・)!! Failed to download !!')
				}

				loop_btn.disabled = false
			}, 'image/png')
		}
	}

	start_btn.addEventListener('click', async () => {
		if (filePicker.value !== '') {
			await process()
		} else {
			set_message('(ﾟ∀ﾟ) No Image Found')
		}
	})

	stop_btn.addEventListener('click', () => {
		onnx_runner.stop_flag = true
	})

	loop_btn.addEventListener('click', () => {
		loop_btn.disabled = true
		filePicker.files = null
		src.width = dest.width
		src.height = dest.height
		src_ctx.drawImage(dest, 0, 0)
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
						const scale_4x = elem.options.namedItem('scale_4x')!
						const scale_2x = elem.options.namedItem('scale_2x')!

						if (elem.selectedIndex == scale_4x.index && !is_swin_unet) {
							elem.selectedIndex = scale_2x.index
							elem.dispatchEvent(new Event('change'))
						}

						scale_4x.disabled = !is_swin_unet
					}
				})

				document.getElementById('scale-comment')!.style.display = is_swin_unet ? 'none' : 'unset'

				document.getElementById('tile-comment')!.style.display =
					currentSettings.model_name.split('.')?.[1] == 'photo' && currentSettings.tile_size < 256 ? 'unset' : 'none'
			})

			elem.dispatchEvent(new Event('change'))
		}
	})

	document.getElementsByName('noise_level').forEach(elem => {
		if (elem instanceof HTMLSelectElement) {
			elem.value = currentSettings.noise.toString()

			elem.addEventListener('change', () => {
				currentSettings.noise = BigInt(elem.value) as any
				save_settings()
			})
		}
	})

	document.getElementsByName('scale').forEach(elem => {
		if (elem instanceof HTMLSelectElement) {
			elem.value = currentSettings.scale.toString()

			elem.addEventListener('change', () => {
				currentSettings.scale = BigInt(elem.value) as any
				save_settings()
			})
		}
	})

	document.getElementsByName('tile_size').forEach(elem => {
		if (elem instanceof HTMLSelectElement) {
			elem.value = currentSettings.tile_size.toString()

			elem.addEventListener('change', () => {
				currentSettings.tile_size = BigInt(elem.value)
				save_settings()

				document.getElementById('tile-comment')!.style.display =
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
				currentSettings.tta_level = BigInt(elem.value) as any
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
