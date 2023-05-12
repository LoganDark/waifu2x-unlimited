/// <reference path="metadata.ts" />

type FloatTensor = ort.TypedTensor<'float32'>

type Batch<BatchSize extends number, Channels extends number> = FloatTensor & {readonly dims: [BatchSize, Channels, number, number]}
type Batch3<BatchSize extends number> = Batch<BatchSize, 3>
type Bitmap<Channels extends number> = Batch<1, Channels>
type Bitmap3 = Bitmap<3>
type Bitmap1 = Bitmap<1>

abstract class Session {
	protected constructor(protected session: ort.InferenceSession) {}

	protected _run = async (feeds: ort.InferenceSession.FeedsType, outputName: string) =>
		(await this.session.run(feeds))[outputName] as FloatTensor
}

class Utils {
	private constructor(
		private _pad: Utils.Padding,
		private _tta: Utils.TTA,
		private _antialias: Utils.Antialias,
		public pad = _pad.pad,
		public pad_alpha = _pad.pad_alpha,
		public tta_split = _tta.split,
		public tta_merge = _tta.merge,
		public antialias = _antialias.antialias
	) {}

	public static load = async () => new this(
		await Utils.Padding.load(),
		await Utils.TTA.load(),
		await Utils.Antialias.load()
	)

	public static sessionOptions: ort.InferenceSession.SessionOptions = {
		executionProviders: ['wasm'],
		executionMode: 'parallel',
		graphOptimizationLevel: 'all'
	}

	public static createSession = (path: string) => ort.InferenceSession.create(path, this.sessionOptions)
	public static createUtilitySession = (name: string) => this.createSession(ModelMetadata.getUtilityPath(name))
	public static createModelSession = (name: string, style: string, scale: number, noise: Utils.NoiseLevel, suffix?: string) =>
		this.createSession(ModelMetadata.getModelPath(name, style, scale, noise, suffix))
}

namespace Utils {
	export type NoiseLevel = -1 | 0 | 1 | 2 | 3
	export type TtaLevel = 0 | 2 | 4

	export class Padding {
		private constructor(
			private _pad: Padding.Pad,
			private _alpha: Padding.AlphaPad,
			public pad = _pad.pad,
			public pad_alpha = _alpha.pad
		) {}

		public static load = async () => new this(
			await Padding.Pad.load(),
			await Padding.AlphaPad.load()
		)
	}

	export namespace Padding {
		export type Amount = {
			left: number,
			right: number,
			top: number,
			bottom: number
		}

		export class Pad extends Session {
			public static load = async () => new this(await Utils.createUtilitySession('pad'))

			public pad = async <B extends number>(image: Batch3<B>, padding: Readonly<Amount>) => await this._run({
				x: image,
				left: new ort.Tensor('int64', BigInt64Array.from([BigInt(padding.left)]), []),
				right: new ort.Tensor('int64', BigInt64Array.from([BigInt(padding.right)]), []),
				top: new ort.Tensor('int64', BigInt64Array.from([BigInt(padding.top)]), []),
				bottom: new ort.Tensor('int64', BigInt64Array.from([BigInt(padding.bottom)]), [])
			}, 'y') as Batch3<B>
		}

		export class AlphaPad extends Session {
			public static load = async () => new this(await Utils.createUtilitySession('alpha_border_padding'))

			public pad = async (rgb: Bitmap3, alpha: Bitmap1, offset: number) => {
				const output = await this._run({
					rgb: new ort.Tensor('float32', rgb.data, rgb.dims.slice(1)),
					alpha: new ort.Tensor('float32', alpha.data, alpha.dims.slice(1)),
					offset: new ort.Tensor('int64', BigInt64Array.from([BigInt(offset)]), [])
				}, 'y') as Bitmap3

				return new ort.Tensor('float32', output.data, [1, ...output.dims]) as Bitmap3
			}
		}
	}

	export class TTA {
		private constructor(
			private _split: TTA.Split,
			private _merge: TTA.Merge,
			public split = _split.split,
			public merge = _merge.merge
		) {}

		public static load = async () => new this(
			await TTA.Split.load(),
			await TTA.Merge.load()
		)
	}

	export namespace TTA {
		export class Split extends Session {
			public static load = async () => new this(await Utils.createUtilitySession('tta_split'))

			public split = async (image: Bitmap3, level: TtaLevel) => await this._run({
				x: image,
				tta_level: new ort.Tensor('int64', BigInt64Array.from([BigInt(level)]), [])
			}, 'y') as Bitmap3
		}

		export class Merge extends Session {
			public static load = async () => new this(await Utils.createUtilitySession('tta_merge'))

			public merge = async (image: Bitmap3, level: TtaLevel) => await this._run({
				x: image,
				tta_level: new ort.Tensor('int64', BigInt64Array.from([BigInt(level)]), [])
			}, 'y') as Bitmap3
		}
	}

	export class Antialias extends Session {
		public static load = async () => new this(await Utils.createUtilitySession('antialias'))

		public antialias = async (image: Bitmap3) => await this._run({x: image}, 'y') as Bitmap3
	}
}

class Model extends Session {
	private constructor(
		session: ort.InferenceSession,
		private _name: string,
		private _style: string,
		private _scale: number,
		private _noise: Utils.NoiseLevel,
		private _suffix: string | null,
		private _context: number
	) {
		super(session)
	}

	public get name(): string { return this._name }

	public get style(): string { return this._style }

	public get scale(): number { return this._scale }

	public get noise(): Utils.NoiseLevel { return this._noise }

	public get suffix(): string | null { return this._suffix }

	public get context() { return this._context }

	public static load = async (name: string, style: string, scale: number, noise: Utils.NoiseLevel, suffix?: string) =>
		new this(await Utils.createModelSession(name, style, scale, noise, suffix), name, style, scale, noise, suffix ?? null, ModelMetadata.models[name].scales.get(scale)!.context)

	public run = async <B extends number>(image: Batch3<B>) => await this._run({x: image}, 'y') as Batch3<B>
}

namespace Model.Cache {
	const cache: {[key: string]: Promise<Model>} = {}

	export const load = (name: string, style: string, scale: number, noise: Utils.NoiseLevel, suffix?: string) => {
		const key = `${name}.${style}:${suffix}:${scale}:${noise}}`
		return cache[key] ?? (cache[key] = Model.load(name, style, scale, noise, suffix))
	}
}
