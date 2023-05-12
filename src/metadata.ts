namespace ModelMetadata {
	export const models: {
		[name: string]: {
			styles: Map<string, {prefersLargeTiles: boolean}>
			scales: Map<number, {context: number}>
		}
	} = {
		swin_unet: {
			styles: new Map([
				['art', {prefersLargeTiles: false}],
				['art_scan', {prefersLargeTiles: true}],
				['photo', {prefersLargeTiles: true}]
			]),
			scales: new Map([
				[1, {context: 10}],
				[2, {context: 16}],
				[4, {context: 12}]
			])
		},
		cunet: {
			styles: new Map([
				['art', {prefersLargeTiles: false}]
			]),
			scales: new Map([
				[1, {context: 28}],
				[2, {context: 18}]
			])
		}
	}

	export let pathBase = 'models/'

	export const getModelBasename = (scale: number, noise: Utils.NoiseLevel): string =>
		(noise === -1 ? `scale${scale}x` : scale === 1 ? `noise${noise}` : `noise${noise}_scale${scale}x`)

	export const getModelPath = (name: string, style: string, scale: number, noise: Utils.NoiseLevel, suffix: string = ''): string =>
		`${pathBase}${name}/${style}/${getModelBasename(scale, noise)}${suffix}.onnx`

	export const getUtilityPath = (name: string): string => `${pathBase}utils/${name}.onnx`
}
