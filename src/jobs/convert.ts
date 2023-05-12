namespace Convert {
	export class GlBlitter {
		private readonly gl: WebGLRenderingContext

		public constructor(private readonly width: number, private readonly height: number) {
			this.gl = new OffscreenCanvas(0, 0).getContext('webgl')!
			if (!this.gl) throw new TypeError('failed to acquire WebGL rendering context')
			this.gl.activeTexture(this.gl.TEXTURE0)
			const texture = this.gl.createTexture()!
			this.gl.bindTexture(this.gl.TEXTURE_2D, texture)
			this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.gl.createFramebuffer()!)
			this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, texture, 0)
			this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, width, height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null)
		}

		public writePixels(source: TexImageSource, x: number, y: number) {
			this.gl.texSubImage2D(this.gl.TEXTURE_2D, 0, x, y, this.gl.RGBA, this.gl.UNSIGNED_BYTE, source)
		}

		public toImageData() {
			const imageData = new ImageData(this.width, this.height)
			this.gl.readPixels(0, 0, this.width, this.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, imageData.data)
			return imageData
		}
	}

	const gl = new OffscreenCanvas(0, 0).getContext('webgl')!

	{
		gl.activeTexture(gl.TEXTURE0)
		const texture = gl.createTexture()!
		gl.bindTexture(gl.TEXTURE_2D, texture)
		gl.bindFramebuffer(gl.FRAMEBUFFER, gl.createFramebuffer()!)
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
	}

	export const readPixels = (source: TexImageSource, dest: Uint8ClampedArray) => {
		const {width, height} = source
		if (dest.length !== width * height * 4) throw new TypeError('invalid destination length')
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source)
		if (source instanceof ImageBitmap) source.close()
		gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, dest)
	}

	export const toRgbAlpha = ({width, height, data}: ImageData) => {
		const pixels = width * height
		const rgb = new Float32Array(pixels * 3)
		const alpha = new Float32Array(pixels)

		let r = 0, g = pixels, b = pixels * 2
		for (let i = 0; i < data.length; i += 4) rgb[r++] = data[i] / 255
		for (let i = 1; i < data.length; i += 4) rgb[g++] = data[i] / 255
		for (let i = 2; i < data.length; i += 4) rgb[b++] = data[i] / 255
		for (let i = 3, a = 0; i < data.length; i += 4) alpha[a++] = data[i] / 255

		return [
			new ort.Tensor('float32', rgb, [1, 3, height, width]) as Bitmap3,
			new ort.Tensor('float32', alpha, [1, 1, height, width]) as Bitmap1
		] as const
	}

	export const stretchAlpha = ({dims, data: alpha1}: Bitmap1) => {
		const [height, width] = dims.slice(2)
		const pixels = width * height
		const alpha3 = new Float32Array(pixels * 3)
		alpha3.set(alpha1, 0)
		alpha3.set(alpha1, pixels)
		alpha3.set(alpha1, pixels * 2)
		return new ort.Tensor('float32', alpha3, [1, 3, height, width]) as Bitmap3
	}

	export const squeezeAlpha = ({dims, data: alpha3}: Bitmap3) => {
		const [height, width] = dims.slice(2)
		const pixels = width * height
		const alpha1 = new Float32Array(pixels)
		let a1 = 0, a2 = pixels, a3 = pixels * 2
		for (let i = 0; i < alpha3.length; i++) alpha1[i] += alpha3[a1++] / 3
		for (let i = 0; i < alpha3.length; i++) alpha1[i] += alpha3[a2++] / 3
		for (let i = 0; i < alpha3.length; i++) alpha1[i] += alpha3[a3++] / 3
		return new ort.Tensor('float32', alpha1, [1, 1, height, width]) as Bitmap1
	}

	export const toRgb = ({width, height, data}: ImageData, bgR = 1, bgG = 1, bgB = 1) => {
		const pixels = width * height
		const rgb = new Float32Array(pixels * 3)
		const alpha = new Float32Array(pixels)

		let a = 0, r = 0, g = pixels, b = pixels * 2
		for (let i = 3; i < data.length; i += 4) alpha[a++] = data[i] / 255
		for (let i = 0, a = 0; i < data.length; i += 4) rgb[r++] = bgR + (data[i] / 255 - bgR) * alpha[a++]
		for (let i = 1, a = 0; i < data.length; i += 4) rgb[g++] = bgG + (data[i] / 255 - bgG) * alpha[a++]
		for (let i = 2, a = 0; i < data.length; i += 4) rgb[b++] = bgB + (data[i] / 255 - bgB) * alpha[a++]

		return new ort.Tensor('float32', rgb, [1, 3, height, width]) as Bitmap3
	}

	export const rgbToImageData = ({dims, data: rgb}: Bitmap3, alpha: Bitmap3 | null = null) => {
		const [height, width] = dims.slice(2)
		const pixels = width * height
		const data = new Uint8ClampedArray(pixels * 4)
		if (!alpha) data.fill(255) // alpha

		let r = 0, g = pixels, b = pixels * 2
		for (let i = 0; i < data.length; i += 4) data[i] = rgb[r++] * 255
		for (let i = 1; i < data.length; i += 4) data[i] = rgb[g++] * 255
		for (let i = 2; i < data.length; i += 4) data[i] = rgb[b++] * 255

		if (alpha) {
			const alphaData = alpha.data
			for (let i = 3, a1 = 0, a2 = pixels, a3 = pixels * 2; i < data.length; i += 4)
				data[i] = (alphaData[a1++] + alphaData[a2++] + alphaData[a3++]) / 3 * 255
		}

		return new ImageData(data, width, height)
	}

	export const bleedEdges = ({dims, data: rgb}: Bitmap3, {data: alpha}: Bitmap1, threshold: number = 0.5) => {
		const width = dims[3]
		const height = dims[2]
		const pixels = width * height

		// SoA (typed arrays) *much* faster than AoS here

		let numEdges = Math.ceil(pixels / 2)
		let numPixels = pixels
		const buffer = new ArrayBuffer(numEdges * 32 + numPixels * 24)
		const edgesX = new Float64Array(buffer, 0, numEdges)
		const edgesY = new Float64Array(buffer, numEdges * 8, numEdges)
		const edgesR = new Float32Array(buffer, numEdges * 16, numEdges)
		const edgesG = new Float32Array(buffer, numEdges * 20, numEdges)
		const edgesB = new Float32Array(buffer, numEdges * 24, numEdges)
		const pixelsX = new Float64Array(buffer, numEdges * 32, numPixels)
		const pixelsY = new Float64Array(buffer, numEdges * 32 + numPixels * 8, numPixels)
		const pixelsI = new Float64Array(buffer, numEdges * 32 + numPixels * 16, numPixels)
		numEdges = 0
		numPixels = 0

		{
			const r = rgb.subarray(0)
			const g = rgb.subarray(pixels)
			const b = rgb.subarray(pixels * 2)

			for (let y = 0, i = 0; y < height; y++) {
				const hasAbove = y > 0
				const hasBelow = y < height - 1

				for (let x = 0; x < width; x++, i++) {
					if (alpha[i] <= threshold) {
						pixelsX[numPixels] = x
						pixelsY[numPixels] = y
						pixelsI[numPixels] = i
						numPixels++
					} else if (
						(x > 0 && alpha[i - 1] <= threshold) || // left
						(x < width - 1 && alpha[i + 1] <= threshold) || // right
						(hasAbove && alpha[i - width] <= threshold) || // above
						(hasBelow && alpha[i + width] <= threshold) // below
					) {
						edgesX[numEdges] = x
						edgesY[numEdges] = y
						edgesR[numEdges] = r[i]
						edgesG[numEdges] = g[i]
						edgesB[numEdges] = b[i]
						numEdges++
					}
				}
			}
		}

		const data = new Float32Array(rgb)

		if (numEdges > 0) {
			const r = data.subarray(0)
			const g = data.subarray(pixels)
			const b = data.subarray(pixels * 2)

			for (let p = 0; p < numPixels; p++) {
				const pixelX = pixelsX[p]
				const pixelY = pixelsY[p]

				let closestDistanceSquared = Infinity
				let closestIndex = 0

				for (let e = 0; e < numEdges; e++) {
					const distanceSquared = (edgesX[e] - pixelX) ** 2 + (edgesY[e] - pixelY) ** 2

					if (distanceSquared < closestDistanceSquared) {
						closestDistanceSquared = distanceSquared
						closestIndex = e
					}
				}

				const i = pixelsI[p]
				r[i] = edgesR[closestIndex]
				g[i] = edgesG[closestIndex]
				b[i] = edgesB[closestIndex]
			}
		}

		return new ort.Tensor('float32', data, [1, 3, height, width]) as Bitmap3
	}

	export const batch = <B extends number>(...images: Bitmap3[] & {length: B}) => {
		const dims = images[0].dims
		const width = dims[3]
		const height = dims[2]

		const batch = new Float32Array(width * height * 3 * images.length)

		let offset = 0
		for (const image of images) {
			if (image.dims[0] > 1) throw new TypeError('image already batched')
			batch.set(image.data, offset)
			offset += image.data.length
		}

		return new ort.Tensor('float32', batch, [images.length, 3, height, width]) as Batch3<B>
	}

	export const unbatch = <B extends number>(batch: Batch3<B>): Bitmap3[] & {length: B} => {
		const dims = batch.dims
		const width = dims[3]
		const height = dims[2]
		const pixels = width * height * 3

		const bitmaps = []

		let offset = 0
		for (let i = 0; i < dims[0]; i++) {
			bitmaps[i] = new ort.Tensor('float32', batch.data.slice(offset, offset += pixels), [1, 3, height, width]) as Bitmap3
		}

		return bitmaps as Bitmap3[] & {length: B}
	}
}
