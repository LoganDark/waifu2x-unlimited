interface ModelConfig {
	scale: bigint
	offset: bigint
	path?: string

	round_tile_size(this: ModelConfig, tile_size: bigint): bigint
}

interface ModelMethods {
	[method: string]: ModelConfig
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

	get_model(arch: string, style: string, method: string) {
		const config = this.models[arch]?.[style]?.[method] ?? null

		if (config !== null) {
			config.path = `models/${arch}/${style}/${method}.onnx`
			return config as Required<ModelConfig>
		}

		return null
	},

	session_options: {
		executionProviders: ['wasm'],
		executionMode: 'parallel',
		graphOptimizationLevel: 'all'
	} as any,

	create: (path: string) => ort.InferenceSession.create(path, Models.session_options),
	get_helper_path: (name: string) => `models/utils/${name}.onnx`,
	create_helper: (name: string) => Models.create(Models.get_helper_path(name)),

	image_into_tensors: (rgb: Float32Array, alpha1: Float32Array, data: Uint8ClampedArray, width: number, height: number) => {
		const len = width * height

		for (let offset = 0; offset < len; offset += width) {
			for (let x = 0; x < width; x++) {
				const i = offset + x
				const i4 = i * 4
				rgb[i] = data[i4] / 255
				rgb[i + len] = data[i4 + 1] / 255
				rgb[i + len * 2] = data[i4 + 2] / 255
				alpha1[i] = data[i4 + 3] / 255
			}
		}
	},

	image_to_tensors: (() => {
		function image_to_tensor(data: Uint8ClampedArray, width: number, height: number, keep_alpha: true): readonly [ort.TypedTensor<'float32'>, ort.TypedTensor<'float32'>, ort.TypedTensor<'float32'>]
		function image_to_tensor(data: Uint8ClampedArray, width: number, height: number, keep_alpha?: false): readonly [ort.TypedTensor<'float32'>, null, null]
		function image_to_tensor(data: Uint8ClampedArray, width: number, height: number, keep_alpha: boolean): readonly [ort.TypedTensor<'float32'>, ort.TypedTensor<'float32'>, ort.TypedTensor<'float32'>] | readonly [ort.TypedTensor<'float32'>, null, null]
		function image_to_tensor(data: Uint8ClampedArray, width: number, height: number, keep_alpha: boolean = false) {
			const len = width * height
			const rgb = new Float32Array(len * 3)
			const alpha1 = new Float32Array(len)

			Models.image_into_tensors(rgb, alpha1, data, width, height)

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
		}

		return image_to_tensor
	})(),

	tensors_into_image_data: (data: Uint8ClampedArray, rgb: Float32Array, alpha3: Float32Array | null, width: number, height: number) => {
		const len = width * height

		for (let y = 0; y < height; y++) {
			const offset = y * width

			for (let x = 0; x < width; x++) {
				const i = offset + x
				const i4 = i * 4
				data[i4] = rgb[i] * 255
				data[i4 + 1] = rgb[i + len] * 255
				data[i4 + 2] = rgb[i + len * 2] * 255
				data[i4 + 3] = (alpha3 ? (alpha3[i] + alpha3[i + len] + alpha3[i + len * 2]) / 3 : 1) * 255
			}
		}
	},

	tensors_to_image: (rgb: Float32Array, alpha3: Float32Array | null, width: number, height: number) => {
		const len = width * height
		const data = new Uint8ClampedArray(len * 4)
		Models.tensors_into_image_data(data, rgb, alpha3, width, height)
		return new ImageData(data, width, height)
	}
}

const BLEND_SIZE = 16n

interface SettingsSnapshot {
	model_name: 'swin_unet.art' | 'swin_unet.photo' | 'cunet'
	noise: -1n | 0n | 1n | 2n | 3n
	noise_antialias: boolean
	scale: 1n | 2n | 4n
	tile_size: bigint
	tile_random: boolean
	tta_level: 0n | 2n | 4n
	detect_alpha: boolean
}

class UtilityModels {
	private constructor(
		private readonly _pad: ort.InferenceSession,
		private readonly _tta_split: ort.InferenceSession,
		private readonly _tta_merge: ort.InferenceSession,
		private readonly _alpha_border_padding: ort.InferenceSession,
		private readonly _antialias: ort.InferenceSession
	) {}

	public static async initialize(session_cache: SessionCache) {
		const pad = await session_cache.get('pad')
		const tta_split = await session_cache.get('tta_split')
		const tta_merge = await session_cache.get('tta_merge')
		const alpha_border_padding = await session_cache.get('alpha_border_padding')
		const antialias = await session_cache.get('antialias')
		return new UtilityModels(pad, tta_split, tta_merge, alpha_border_padding, antialias)
	}

	public async pad(rgb: ort.TypedTensor<'float32'>, padding: Padding) {
		const output = await this._pad.run({
			x: rgb,
			left: new ort.Tensor('int64', BigInt64Array.from([padding.left]), []),
			right: new ort.Tensor('int64', BigInt64Array.from([padding.right]), []),
			top: new ort.Tensor('int64', BigInt64Array.from([padding.top]), []),
			bottom: new ort.Tensor('int64', BigInt64Array.from([padding.bottom]), [])
		})

		return output.y as ort.TypedTensor<'float32'>
	}

	public async tta_split(rgb: ort.TypedTensor<'float32'>, level: bigint) {
		const output = await this._tta_split.run({
			x: rgb,
			tta_level: new ort.Tensor('int64', BigInt64Array.from([level]), [])
		})

		return output.y as ort.TypedTensor<'float32'>
	}

	public async tta_merge(rgb: ort.TypedTensor<'float32'>, level: bigint) {
		const output = await this._tta_merge.run({
			x: rgb,
			tta_level: new ort.Tensor('int64', BigInt64Array.from([level]), [])
		})

		return output.y as ort.TypedTensor<'float32'>
	}

	public async pad_alpha_border(rgb: ort.TypedTensor<'float32'>, alpha1: ort.TypedTensor<'float32'>, offset: bigint) {
		const result = await this._alpha_border_padding.run({
			rgb: new ort.Tensor('float32', rgb.data, rgb.dims.slice(1)),
			alpha: new ort.Tensor('float32', alpha1.data, alpha1.dims.slice(1)),
			offset: new ort.Tensor('int64', BigInt64Array.from([offset]), [])
		})

		const out_rgb = result.y as ort.TypedTensor<'float32'>
		return new ort.Tensor('float32', out_rgb.data, [1, ...out_rgb.dims])
	}

	public async antialias(rgb: ort.TypedTensor<'float32'>) {
		const result = await this._antialias.run({x: rgb})
		return result.y as ort.TypedTensor<'float32'>
	}
}

class SessionCache {
	private cache: Map<Required<ModelConfig> | string, Promise<ort.InferenceSession>> = new Map()

	public constructor() {}

	public get(specifier: Required<ModelConfig> | string) {
		const cache = this.cache
		const existing = cache.get(specifier)

		if (!existing) {
			const session = Models.create(typeof specifier === 'string' ? Models.get_helper_path(specifier) : specifier.path)
			cache.set(specifier, session)
			return session
		}

		return existing
	}

	public delete(config: Required<ModelConfig>) {
		this.cache.delete(config)
	}
}

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
	public readonly output_height: bigint
	public readonly output_width: bigint
	public readonly input_offset: bigint
	public readonly input_blend_size: bigint
	public readonly input_tile_step: bigint
	public readonly output_tile_step: bigint
	public readonly output_tile_size: bigint
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

		this.output_height = input_height * scale
		this.output_width = input_width * scale
		this.input_offset = (offset + scale - 1n) / scale
		this.input_blend_size = (blend_size + scale - 1n) / scale
		this.input_tile_step = tile_size - (this.input_offset * 2n + this.input_blend_size)
		this.output_tile_step = this.input_tile_step * scale
		this.output_tile_size = tile_size * scale - offset * 2n

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
			out_w: this.tile_size * this.scale - this.offset * 2n,
			out_h: this.tile_size * this.scale - this.offset * 2n
		}
	}
}

class SeamBlender {
	public static Factory = class SeamBlenderFactory {
		private constructor(private readonly create_seam_blending_filter: ort.InferenceSession) {}

		public static async initialize() {
			return new SeamBlenderFactory(await Models.create_helper('create_seam_blending_filter'))
		}

		private static async create_tile_filter(create_seam_blending_filter: ort.InferenceSession, scale: bigint, offset: bigint, tile_size: bigint) {
			const result = await create_seam_blending_filter.run({
				scale: new ort.Tensor('int64', BigInt64Array.from([scale]), []),
				offset: new ort.Tensor('int64', BigInt64Array.from([offset]), []),
				tile_size: new ort.Tensor('int64', BigInt64Array.from([tile_size]), [])
			})

			return result.y as ort.TypedTensor<'float32'>
		}

		public async produce(params: TilingParameters) {
			const {output_buffer_h, output_buffer_w, scale, offset, tile_size} = params
			const image_pixels = new ort.Tensor('float32', new Float32Array(output_buffer_h * output_buffer_w * 3), [3, output_buffer_h, output_buffer_w])
			const image_weights = new ort.Tensor('float32', new Float32Array(output_buffer_h * output_buffer_w * 3), [3, output_buffer_h, output_buffer_w])
			const tile_filter = await SeamBlenderFactory.create_tile_filter(this.create_seam_blending_filter, scale, offset, tile_size)
			const tile_pixels = new ort.Tensor('float32', new Float32Array(tile_filter.data.length), tile_filter.dims)
			return new SeamBlender(params, image_pixels, image_weights, tile_filter, tile_pixels)
		}
	}

	private constructor(
		private readonly params: TilingParameters,
		private readonly image_pixels: ort.TypedTensor<'float32'>,
		private readonly image_weights: ort.TypedTensor<'float32'>,
		private readonly tile_filter: ort.TypedTensor<'float32'>,
		private readonly tile_pixels: ort.TypedTensor<'float32'>
	) {}

	public blend(tile: ort.TypedTensor<'float32'>, tile_x: bigint, tile_y: bigint) {
		const [, tile_h, tile_w] = this.tile_pixels!.dims
		const tile_size = tile_w * tile_h

		const [, image_h, image_w] = this.image_pixels!.dims
		const image_size = image_w * image_h

		const step_size = this.params!.output_tile_step
		const output_tile_x = Number(step_size * tile_x),
		      output_tile_y = Number(step_size * tile_y)

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

type SeamBlenderFactory = Awaited<ReturnType<typeof SeamBlender.Factory.initialize>>

class TileMapper {
	public static Factory = class TileMapperFactory {
		private constructor(
			private readonly session_cache: SessionCache,
			private readonly utils: UtilityModels,
			private readonly blender_factory: SeamBlenderFactory
		) {}

		public static async initialize(session_cache: SessionCache) {
			const utils = await UtilityModels.initialize(session_cache)
			const blender_factory = await SeamBlender.Factory.initialize()
			return new TileMapperFactory(session_cache, utils, blender_factory)
		}

		public async create(tile_size: bigint, scale: bigint, offset: bigint, image: ImageData, preprocess_alpha: boolean) {
			const {session_cache, utils, blender_factory} = this

			const params = new TilingParameters(BigInt(image.width), BigInt(image.height), scale, offset, tile_size)

			if (preprocess_alpha) {
				const [rgb, alpha1, alpha3] = Models.image_to_tensors(image.data, image.width, image.height, true)
				const rgb_padded = await utils.pad(await utils.pad_alpha_border(rgb, alpha1, params.offset), params.padding)
				const alpha3_padded = await utils.pad(alpha3, params.padding)
				image = Models.tensors_to_image(rgb_padded.data, alpha3_padded.data, rgb_padded.dims[3], rgb_padded.dims[2])
			} else {
				const [rgb] = Models.image_to_tensors(image.data, image.width, image.height)
				const rgb_padded = await utils.pad(rgb, params.padding)
				image = Models.tensors_to_image(rgb_padded.data, null, rgb_padded.dims[3], rgb_padded.dims[2])
			}

			const input_canvas = new OffscreenCanvas(image.width, image.height)
			const input_ctx = input_canvas.getContext('2d', {willReadFrequently: true})!
			input_ctx.putImageData(image, 0, 0)

			const output_canvas = new OffscreenCanvas(Number(params.output_width), Number(params.output_height))
			const output_ctx = output_canvas.getContext('2d', {willReadFrequently: true})!

			const color_blender = await blender_factory.produce(params),
			      alpha_blender = await blender_factory.produce(params)

			const tiles = []
			for (let row = 0n; row < params.num_h_blocks; row++) {
				for (let col = 0n; col < params.num_w_blocks; col++) {
					tiles.push([row, col] as const)
				}
			}

			return new TileMapper(session_cache, utils, params, input_canvas, input_ctx, output_canvas, output_ctx, color_blender, alpha_blender, tiles)
		}
	}

	private constructor(
		private readonly session_cache: SessionCache,
		private readonly utils: UtilityModels,
		private readonly params: TilingParameters,
		private readonly input_canvas: OffscreenCanvas,
		private readonly input_ctx: OffscreenCanvasRenderingContext2D,
		private readonly output_canvas: OffscreenCanvas,
		private readonly output_ctx: OffscreenCanvasRenderingContext2D,
		private readonly color_blender: SeamBlender,
		private readonly alpha_blender: SeamBlender,
		private readonly unmapped_tiles: (readonly [bigint, bigint])[]
	) {
		this.image_data_array = new Uint8ClampedArray(Number(this.params.output_tile_size) ** 2 * 4)
		this.image_data = new ImageData(this.image_data_array, Number(this.params.output_tile_size), Number(this.params.output_tile_size))
	}

	private readonly image_to_tiles: WeakMap<ImageData, readonly [bigint, bigint]> = new WeakMap()
	private readonly mapped_tiles: {[K: number]: boolean} = []

	private readonly image_data_array: Uint8ClampedArray
	private readonly image_data: ImageData

	public get_utils() {
		return this.utils
	}

	public get_session_cache() {
		return this.session_cache
	}

	public get_parameters() {
		return this.params
	}

	public tiles_remaining() {
		return this.unmapped_tiles.length
	}

	private obtain_tile_image(row: bigint, col: bigint) {
		const {in_x, in_y, in_w, in_h} = this.params.get_tile_params(row, col)
		const image = this.input_ctx.getImageData(Number(in_x), Number(in_y), Number(in_w), Number(in_h))
		this.image_to_tiles.set(image, [row, col] as const)
		return image
	}

	public take_next() {
		const tile = this.unmapped_tiles.shift() ?? null
		return tile ? this.obtain_tile_image(...tile) : null
	}

	public take_random() {
		const tile = this.unmapped_tiles.splice(Math.floor(Math.random() * this.unmapped_tiles.length), 1).shift() ?? null
		return tile ? this.obtain_tile_image(...tile) : null
	}

	public cancel_submission(tile: ImageData) {
		const coords = this.image_to_tiles.get(tile)!
		if (!this.image_to_tiles.delete(tile)) return
		this.unmapped_tiles.unshift(coords)
	}

	public cancel_all_submissions() {
		const tiles = this.unmapped_tiles
		tiles.splice(0, tiles.length)

		const {num_h_blocks, num_w_blocks} = this.params
		for (let row = 0n; row < num_h_blocks; row++) {
			for (let col = 0n; col < num_w_blocks; col++) {
				if (!this.mapped_tiles[Number(row * num_w_blocks + col)]) {
					tiles.push([row, col] as const)
				}
			}
		}
	}

	public submit_mapped_tile(tile: ImageData, color: ort.TypedTensor<'float32'>, alpha: ort.TypedTensor<'float32'> | null) {
		const coords = this.image_to_tiles.get(tile)!
		if (!this.image_to_tiles.delete(tile)) throw new TypeError('invalid tile submission')

		const [row, col] = coords
		const {out_x, out_y, out_w, out_h} = this.params.get_tile_params(row, col)

		this.mapped_tiles[Number(row * this.params.num_w_blocks + col)] = true

		color = this.color_blender.blend(color, col, row)
		if (alpha) alpha = this.alpha_blender.blend(alpha, col, row)

		Models.tensors_into_image_data(this.image_data_array, color.data, alpha?.data ?? null, Number(out_w), Number(out_h))
		this.output_ctx.putImageData(this.image_data, Number(out_x), Number(out_y))
		return [out_x, out_y, out_w, out_h] as const
	}

	public get_output_canvas() {
		return this.output_canvas
	}
}

class SolidColorTensor {
	private readonly len: number
	private readonly rgb: ort.TypedTensor<'float32'>
	private readonly alpha3: ort.TypedTensor<'float32'>

	public constructor(width: number, height: number) {
		const len = this.len = width * height
		this.rgb = new ort.Tensor('float32', new Float32Array(len * 3), [1, 3, height, width])
		this.alpha3 = new ort.Tensor('float32', new Float32Array(len * 3), [1, 3, height, width])
	}

	public color(r: number, g: number, b: number, a: number) {
		const len = this.len
		this.rgb.data.fill(r, 0, len)
		this.rgb.data.fill(g, len, len * 2)
		this.rgb.data.fill(b, len * 2, len * 3)
		this.alpha3.data.fill(a)

		return [this.rgb, this.alpha3] as const
	}
}

const onnx_runner = {
	stop_flag: false,
	running: false,

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

	async tiled_render(
		session_cache: SessionCache,
		image_data: ImageData,
		model_config: Required<ModelConfig>,
		alpha_config: Required<ModelConfig> | null,
		settings: SettingsSnapshot,
		output_canvas: HTMLCanvasElement
	) {
		this.stop_flag = false

		if (this.running) {
			console.log('Already running')
			return
		}

		try {
			this.running = true

			const tile_size = model_config.round_tile_size(settings.tile_size)

			const tile_mapper_factory = await TileMapper.Factory.initialize(session_cache)
			const tile_mapper = await tile_mapper_factory.create(tile_size, model_config.scale, model_config.offset, image_data, alpha_config !== null)

			const has_alpha = alpha_config != null

			const tile_mapper_output = tile_mapper.get_output_canvas()
			output_canvas.width = tile_mapper_output.width
			output_canvas.height = tile_mapper_output.height

			const dirty_regions: ReturnType<typeof tile_mapper.submit_mapped_tile>[] = []
			const output_ctx = output_canvas.getContext('2d')!

			const update_output_canvas = () => {
				for (const region of dirty_regions.splice(0)) {
					const [x, y, w, h] = region.map(Number)
					output_ctx.drawImage(tile_mapper_output, x, y, w, h, x, y, w, h)
				}
			}

			const output_tile_size = Number(tile_size * model_config.scale - model_config.offset * 2n)
			const solid_color = new SolidColorTensor(output_tile_size, output_tile_size)

			// noinspection JSAssignmentUsedAsCondition
			for (let tile_image; tile_image = tile_mapper.take_next();) {
				const color = this.is_solid_color(tile_image.data, has_alpha)
				if (color) dirty_regions.push(tile_mapper.submit_mapped_tile(tile_image, ...solid_color.color(...color)))
			}

			tile_mapper.cancel_all_submissions()
			update_output_canvas()

			let next_update = performance.now() + 100
			let animation_frame = null

			const update_during_animation_frame = () => {
				const now = performance.now()
				if (now >= next_update && dirty_regions.length > 0) {
					update_output_canvas()
					const now2 = performance.now()
					next_update = now2 + Math.max((now2 - now) * 9, 150)
				}

				animation_frame = requestAnimationFrame(update_during_animation_frame)
			}

			animation_frame = requestAnimationFrame(update_during_animation_frame)

			try {
				const utils = tile_mapper.get_utils()
				const model = await session_cache.get(model_config)
				const alpha_model = has_alpha ? await session_cache.get(alpha_config) : null

				while (true) {
					if (this.stop_flag) break

					const tile_image = settings.tile_random ? tile_mapper.take_random() : tile_mapper.take_next()
					if (tile_image === null) break

					let [tile, , tile_alpha] = Models.image_to_tensors(tile_image.data, tile_image.width, tile_image.height, has_alpha)

					if (settings.tta_level > 0) tile = await utils.tta_split(tile, settings.tta_level)

					if (this.stop_flag) break

					if (settings.noise_antialias) {
						tile = await utils.antialias(tile)
						if (tile_alpha) tile_alpha = await utils.antialias(tile_alpha)
					}

					const output = await model.run({x: tile})
					tile = output.y as ort.TypedTensor<'float32'>

					if (this.stop_flag) break

					const alpha_output = alpha_model && tile_alpha ? await alpha_model.run({x: tile_alpha}) : null
					if (alpha_output) tile_alpha = alpha_output.y as ort.TypedTensor<'float32'>

					if (this.stop_flag) break

					if (settings.tta_level > 0) tile = await utils.tta_merge(tile, settings.tta_level)

					dirty_regions.push(tile_mapper.submit_mapped_tile(tile_image, tile, tile_alpha))
				}
			} finally {
				cancelAnimationFrame(animation_frame)
				update_output_canvas()
			}
		} finally {
			this.running = false
		}
	}
}

const currentSettings: SettingsSnapshot = {
	model_name: 'swin_unet.art',
	noise: 0n,
	noise_antialias: false,
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
	ort.env.wasm.numThreads = navigator.hardwareConcurrency
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

	async function set_input_image(file: File) {
		const feedbackWasDisabled = loop_btn.disabled

		start_btn.disabled = true
		loop_btn.disabled = true

		const bitmap = await createImageBitmap(file)
		src.width = bitmap.width
		src.height = bitmap.height
		src_ctx.drawImage(bitmap, 0, 0)
		src.style.height = '128px'
		src.style.width = 'auto'
		filename = file.name

		start_btn.disabled = false
		loop_btn.disabled = feedbackWasDisabled
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
			//set_message('( ・∀・)b')
		} else {
			clear_input_image()
			//set_message('(ﾟ∀ﾟ)', 1)
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
	const session_cache = new SessionCache()

	async function process() {
		if (onnx_runner.running) {
			console.log('Already running')
			return
		}

		const settings = {...currentSettings}

		const [arch, style] = settings.model_name.split('.')
		let method: string

		if (settings.scale == 1n) {
			if (settings.noise == -1n) {
				//set_message('(・A・) No Noise Reduction selected!')
				method = 'scale1x'
			} else {
				method = `noise${settings.noise}`
			}
		} else {
			if (settings.noise == -1n) {
				method = `scale${settings.scale}x`
			} else {
				method = `noise${settings.noise}_scale${settings.scale}x`
			}
		}

		const config = Models.get_model(arch, style, method)

		if (config == null) {
			//set_message('(・A・) Model Not found!')
			return
		}

		const image_data = src_ctx.getImageData(0, 0, src.width, src.height)
		const has_alpha = !settings.detect_alpha ? false : onnx_runner.has_transparency(image_data.data)
		const alpha_config = has_alpha ? Models.get_model(arch, style, /scale\d+x/.exec(method)?.[0] ?? 'scale1x') : null

		if (has_alpha && !alpha_config) {
			//set_message('(・A・) Alpha Model Not found!')
			return
		}

		//set_message('(・∀・)φ ... ', -1)

		loop_btn.disabled = true

		await onnx_runner.tiled_render(session_cache, image_data, config, alpha_config, settings, dest)

		if (!onnx_runner.stop_flag) {
			dest.toBlob((blob) => {
				if (blob != null) {
					const url = URL.createObjectURL(blob)
					const download_filename = (filename.split(/(?=\.[^.]+$)/))[0] + '_waifu2x_' + method + '.png'
					//set_message(`( ・∀・)つ　<a href="${url}" download="${download_filename}">Download</a>`, -1, true)
				} else {
					//set_message('(・A・)!! Failed to download !!')
				}

				loop_btn.disabled = false
			}, 'image/png')
		}
	}

	start_btn.addEventListener('click', async () => {
		if (filePicker.value !== '') {
			await process()
		} else {
			//set_message('(ﾟ∀ﾟ) No Image Found')
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

	document.getElementsByName('noise_antialias').forEach(elem => {
		if (elem instanceof HTMLInputElement && elem.type === 'checkbox') {
			elem.checked = currentSettings.noise_antialias

			elem.addEventListener('change', () => {
				currentSettings.noise_antialias = elem.checked
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
}
