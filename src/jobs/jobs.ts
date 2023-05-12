/// <reference path="convert.ts" />

namespace Jobs {
	import InitParameters = UserInterface.InitParameters
	import LiveParameters = UserInterface.LiveParameters
	import GlBlitter = Convert.GlBlitter

	export type State = 'stop' | 'paused' | 'running'

	export interface Init {
		utils: Utils
		initParams: InitParameters
		liveParams: LiveParameters

		getState: () => State
		getNextState: () => Promise<State>

		reportCallback: (report: Report) => void
	}

	export type Report = StartedReport | PausedReport | UnpausedReport | TileStartedReport | TileCompletedReport | CompletedReport | AbortedReport | ErroredReport

	export interface StartedReport {
		type: 'start'
	}

	export interface PausedReport {
		type: 'paused'
	}

	export interface UnpausedReport {
		type: 'unpaused'
	}

	export interface TileStartedReport {
		type: 'tile-started'
		tilesTotal: number
		tilesCompleted: number
		pixelsTotal: number
		pixelsCompleted: number
		pixelsStarted: number
	}

	export interface TileCompletedReport {
		type: 'tile-completed'
		tilesTotal: number
		tilesCompleted: number
		pixelsTotal: number
		pixelsCompleted: number
		pixelsJustCompleted: number
	}

	export interface CompletedReport {
		type: 'completed'
		value: ImageBitmap
	}

	export interface AbortedReport {
		type: 'aborted'
	}

	export interface ErroredReport {
		type: 'errored'
		error: any
	}

	interface Store extends Init {
		stop: boolean
		timer: Timer
		yield: () => Promise<void>
		wrap: <T>(promise: Promise<T>) => Promise<T>
		output?: GlBlitter
	}

	export const spawn = (init: Init) => new Promise<ImageBitmap | null>((accept, reject) => {
		if (init.getState() === 'stop') {
			init.reportCallback({type: 'aborted'})
			accept(null)
			return
		}

		const store: Store = {
			...init,

			stop: false,
			timer: new Timer(),

			yield: () => new Promise(async accept => {
				if (store.stop) return
				let state = store.getState()

				if (state === 'paused') {
					store.reportCallback({type: 'paused'})
					while (state === 'paused') state = await store.getNextState()
					if (state === 'stop' || store.stop) return
					store.reportCallback({type: 'unpaused'})
				}

				if (state !== 'stop' && !store.stop) accept()
			}),

			wrap: <T>(promise: Promise<T>) => new Promise<T>(async (accept, reject) => {
				await store.yield()
				await promise.then(async (value) => {
					await store.yield()
					accept(value)
				}, async (value) => {
					await store.yield()
					reject(value)
				})
			}),

			output: undefined
		}

		const onFinished = (report: CompletedReport | AbortedReport | ErroredReport) => {
			console.info(`${'―'.repeat(4)} unlimited:waifu2x job ${report.type === 'aborted' ? 'aborted' : report.type === 'errored' ? 'FAILED' : 'completed'} ${'―'.repeat(91 + (report.type === 'aborted' ? 2 : report.type === 'errored' ? 3 : 0))}`)
			console.info(`· Input: ${store.initParams.image.width}x${store.initParams.image.height} (${store.initParams.image.width * store.initParams.image.height}px)`)
			console.info(`· Output: ${store.initParams.image.width * store.initParams.scale}x${store.initParams.image.height * store.initParams.scale} (${store.initParams.image.width * store.initParams.scale * store.initParams.image.height * store.initParams.scale}px)`)
			console.info(`· Model: ${store.initParams.model.name}.${store.initParams.model.style}`)
			console.info(`· Denoise: ${store.initParams.model.noise}`)
			console.info(`· Scale: ${store.initParams.scale}`)
			console.info(`· Tile size: ${store.initParams.tileSize}`)
			console.info(`· TTA level: ${store.initParams.ttaLevel}`)
			console.info(`· Alpha: ${store.initParams.alphaChannel}`)
			console.info(`· Threads: ${!!SharedArrayBuffer ? ort.env.wasm.numThreads : '1 (no SharedArrayBuffer)'}`)
			console.info('\n')
			store.timer.printSummary()
			console.info('―'.repeat(128))
			return report
		}

		// avoid ABA
		store.getNextState().then(async function watchForStop(newState) {
			if (newState === 'stop') {
				if (store.stop) return
				store.stop = true
				store.reportCallback(onFinished({type: 'aborted'}))
				accept(null)
			} else {
				store.getNextState().then(watchForStop)
			}
		})

		run(store).then(
			output => {
				store.stop = true
				store.reportCallback(onFinished({type: 'completed', value: output}))
				accept(output)
			},
			error => {
				store.stop = true
				store.reportCallback(onFinished({type: 'errored', error}))
				reject(error)
			}
		)
	})

	function collectTiles(width: number, height: number, tileSize: number) {
		const tiles = []

		for (let tileY = 0; tileY < height; tileY += tileSize) {
			for (let tileX = 0; tileX < width; tileX += tileSize) {
				tiles.push([tileX, tileY] as const)
			}
		}

		return tiles
	}

	function calculateTileMetrics(width: number, height: number, tileX: number, tileY: number, tileSize: number, context: number) {
		const tileWidth = Math.min(tileSize, width - tileX)
		const tileHeight = Math.min(tileSize, height - tileY)
		const tileWidthAdjusted = Math.ceil(tileWidth / 4) * 4
		const tileHeightAdjusted = Math.ceil(tileHeight / 4) * 4

		const tilePixels = (tileWidthAdjusted + context * 2) * (tileHeightAdjusted + context * 2)
		const tilePadding = {
			left: Math.max(context - tileX, 0),
			right: Math.max(tileX + tileWidthAdjusted + context - width, 0),
			top: Math.max(context - tileY, 0),
			bottom: Math.max(tileY + tileHeightAdjusted + context - height, 0)
		}

		const captureX = tileX - context + tilePadding.left
		const captureY = tileY - context + tilePadding.top
		const captureW = Math.min(tileX + tileWidthAdjusted + context, width) - captureX
		const captureH = Math.min(tileY + tileHeightAdjusted + context, height) - captureY

		return {tileWidth, tileHeight, tileWidthAdjusted, tileHeightAdjusted, tilePixels, tilePadding, captureX, captureY, captureW, captureH}
	}

	function calculateTotalPixels(tiles: any[], width: number, height: number, tileSize: number, context: number) {
		let pixelsTotal = 0

		for (const [tileX, tileY] of tiles)
			pixelsTotal += calculateTileMetrics(width, height, tileX, tileY, tileSize, context).tilePixels

		return pixelsTotal
	}

	const run = async (store: Store) => {
		const {
			utils,
			initParams: {
				image: input,
				image: {width: inputWidth, height: inputHeight},
				output,
				scale,
				tileSize,
				alphaChannel,
				alphaThreshold,
				model,
				model: {context},
				antialias,
				ttaLevel
			},
			liveParams,
			timer,
			wrap
		} = store

		const outputWidth = inputWidth * scale
		const outputHeight = inputHeight * scale

		timer.push('run')
		timer.push('canvas')

		timer.push('getContext')
		const ctx = output.getContext('2d')!
		timer.transition('clear')
		ctx.clearRect(0, 0, outputWidth, outputHeight)

		timer.transition('backdrop')
		ctx.filter = `brightness(75%) blur(${Math.max(outputWidth, outputHeight) * 0.01}px)`
		ctx.drawImage(input, 0, 0, outputWidth, outputHeight)
		ctx.filter = 'none'
		timer.pop()

		timer.transition('glBlitter')
		store.output = new GlBlitter(outputWidth, outputHeight)

		timer.transition('tiledRender')

		timer.push('collectTiles')
		const tiles = collectTiles(inputWidth, inputHeight, tileSize)
		const tilesTotal = tiles.length
		let tilesCompleted = 0

		timer.push('calculateTotalPixels')
		const pixelsTotal = calculateTotalPixels(tiles, inputWidth, inputHeight, tileSize, context)
		let pixelsCompleted = 0

		timer.transition('newScratch')
		const scratch = new Uint8ClampedArray((Math.ceil(tileSize / 4) * 4 + context * 2) ** 2 * 4)

		await store.yield()

		timer.pop()

		// TODO move these stages into separate functions, maybe support batching
		while (tiles.length > 0) {
			timer.push('tile')
			timer.push('pick')

			let tileX = 0, tileY = 0

			const focus = liveParams.tileFocus()
			if (focus) {
				timer.push('focus')
				const [focusX, focusY] = focus
				let closestDistanceSquared = Infinity
				let closestTileIndex = 0

				for (let i = 0; i < tiles.length; i++) {
					const [tileX, tileY] = tiles[i]
					const distanceSquared = (tileX + tileSize / 2 - focusX / scale) ** 2 + (tileY + tileSize / 2 - focusY / scale) ** 2

					if (distanceSquared < closestDistanceSquared) {
						closestDistanceSquared = distanceSquared
						closestTileIndex = i
					}
				}

				[tileX, tileY] = tiles.splice(closestTileIndex, 1)[0]
				timer.pop()
			} else {
				[tileX, tileY] = liveParams.tileRandom()
					? tiles.splice(Math.floor(Math.random() * tiles.length), 1)[0]
					: tiles.shift()!
			}

			timer.transition('calculateTileMetrics')
			const metrics = calculateTileMetrics(inputWidth, inputHeight, tileX, tileY, tileSize, context)

			timer.transition('reportCallbackStarted')
			store.reportCallback({
				type: 'tile-started',
				tilesTotal: tilesTotal,
				tilesCompleted: tilesCompleted,
				pixelsTotal: pixelsTotal,
				pixelsCompleted: pixelsCompleted,
				pixelsStarted: metrics.tilePixels
			})

			timer.transition('indicator')

			const tileMinDim = Math.min(metrics.tileWidth, metrics.tileHeight)
			ctx.strokeStyle = 'deepskyblue'
			ctx.lineWidth = Math.floor(Math.min(Math.max(tileMinDim / 10 * scale, 2), tileMinDim / 3 * scale))
			ctx.strokeRect(
				tileX * scale + ctx.lineWidth / 2,
				tileY * scale + ctx.lineWidth / 2,
				metrics.tileWidth * scale - ctx.lineWidth,
				metrics.tileHeight * scale - ctx.lineWidth
			)

			timer.transition('readPixels')

			const capture = new ImageData(scratch.subarray(0, metrics.captureW * metrics.captureH * 4), metrics.captureW, metrics.captureH)
			Convert.readPixels(await wrap(createImageBitmap(input, metrics.captureX, metrics.captureY, metrics.captureW, metrics.captureH, {premultiplyAlpha: 'none'})), capture.data)

			timer.transition('process')

			let uncropped: ImageData

			if (alphaChannel) {
				timer.push('toRgbAlpha')
				let [rgb, alpha1] = Convert.toRgbAlpha(capture)

				timer.transition('bleedEdges')
				rgb = Convert.bleedEdges(rgb, alpha1, alphaThreshold)

				timer.transition('stretchAlpha')
				let alpha3 = Convert.stretchAlpha(alpha1)

				timer.transition('pad')
				timer.push('rgb')
				rgb = await wrap(utils.pad(rgb, metrics.tilePadding))
				timer.transition('alpha')
				alpha3 = await wrap(utils.pad(alpha3, metrics.tilePadding))
				timer.pop()

				if (ttaLevel > 0) {
					timer.transition('ttaSplit')
					timer.push('rgb')
					rgb = await wrap(utils.tta_split(rgb, ttaLevel))
					timer.transition('alpha')
					alpha3 = await wrap(utils.tta_split(alpha3, ttaLevel))
					timer.pop()
				}

				if (antialias) {
					timer.transition('antialias')
					timer.push('rgb')
					rgb = await wrap(utils.antialias(rgb))
					timer.transition('alpha')
					alpha3 = await wrap(utils.antialias(alpha3))
					timer.pop()
				}

				timer.transition('model')

				// TTA does its own batching
				if (ttaLevel === 0) {
					timer.push('batch')
					let batch = Convert.batch(rgb, alpha3)
					timer.transition('run')
					batch = await wrap(model.run(batch))
					timer.transition('unbatch')
					;[rgb, alpha3] = Convert.unbatch(batch)
					timer.pop()
				} else {
					timer.push('rgb')
					rgb = await wrap(model.run(rgb))
					timer.transition('alpha')
					alpha3 = await wrap(model.run(alpha3))
					timer.pop()
				}

				if (ttaLevel > 0) {
					timer.transition('ttaMerge')
					timer.push('rgb')
					rgb = await wrap(utils.tta_merge(rgb, ttaLevel))
					timer.transition('alpha')
					alpha3 = await wrap(utils.tta_merge(alpha3, ttaLevel))
					timer.pop()
				}

				timer.transition('rgbToImageData')
				uncropped = Convert.rgbToImageData(rgb, alpha3)
				timer.pop()
			} else {
				timer.push('toRgb')
				let rgb = Convert.toRgb(capture)

				timer.transition('pad')
				rgb = await wrap(utils.pad(rgb, metrics.tilePadding))

				if (ttaLevel > 0) {
					timer.transition('ttaSplit')
					rgb = await wrap(utils.tta_split(rgb, ttaLevel))
				}

				if (antialias) {
					timer.transition('antialias')
					rgb = await wrap(utils.antialias(rgb))
				}

				timer.transition('model')
				rgb = await wrap(model.run(rgb))

				if (ttaLevel > 0) {
					timer.transition('ttaMerge')
					rgb = await wrap(utils.tta_merge(rgb, ttaLevel))
				}

				timer.transition('rgbToImageData')
				uncropped = Convert.rgbToImageData(rgb)
				timer.pop()
			}

			timer.transition('crop')

			// crop out all the padding and model-specific artifacts
			const regionDiffX = uncropped.width - metrics.tileWidthAdjusted * scale
			const regionDiffY = uncropped.height - metrics.tileHeightAdjusted * scale
			const cropped = await wrap(createImageBitmap(uncropped, regionDiffX / 2, regionDiffY / 2, metrics.tileWidth * scale, metrics.tileHeight * scale, {premultiplyAlpha: 'none'}))

			timer.transition('writePixels')
			store.output.writePixels(cropped, tileX * scale, tileY * scale)

			timer.transition('clearRect')
			ctx.clearRect(tileX * scale, tileY * scale, metrics.tileWidth * scale, metrics.tileHeight * scale)
			timer.transition('drawImage')
			ctx.drawImage(cropped, tileX * scale, tileY * scale)

			timer.transition('updateCounters')
			tilesCompleted++
			pixelsCompleted += metrics.tilePixels

			timer.transition('reportCallbackCompleted')
			store.reportCallback({type: 'tile-completed', tilesTotal, tilesCompleted, pixelsTotal, pixelsCompleted, pixelsJustCompleted: metrics.tilePixels})
			timer.pop()

			timer.pop()
		}

		timer.transition('toImageData')
		const outputData = store.output.toImageData()

		timer.transition('outputBitmap')
		const outputBitmap = await wrap(createImageBitmap(outputData, {premultiplyAlpha: 'none'}))
		timer.pop()

		timer.pop()

		return outputBitmap
	}
}
