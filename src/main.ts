/// <reference path="models.ts" />
/// <reference path="interface/interface.ts" />
/// <reference path="interface/message.ts" />
/// <reference path="interface/jobStatus.ts" />
/// <reference path="jobs/jobs.ts" />
/// <reference path="metadata.ts" />

(async () => {
	ort.env.wasm.numThreads = navigator.hardwareConcurrency
	ort.env.wasm.simd = true
	ort.env.wasm.proxy = true

	// noinspection ES6MissingAwait
	UserInterface.setMessageWorking('Loading utility models')

	const utils = await Utils.load()

	await UserInterface.setStartCallback(async (initParams, liveParams) => {
		const modelSuffix = `${initParams.model.name}_${initParams.model.style}_${ModelMetadata.getModelBasename(initParams.scale, initParams.model.noise)}`

		const mapState = (state: ReturnType<typeof UserInterface.getState>): Jobs.State => {
			if (state === 'preinit' || state === 'busy') {
				return 'stop'
			} else {
				return state
			}
		}

		let status: JobStatusIndicator | null = null

		const outputBitmap = await Jobs.spawn({
			utils,
			initParams,
			liveParams,
			getState: () => mapState(UserInterface.getState()),
			getNextState: () => UserInterface.getNextState().then(mapState),
			reportCallback: async (report: Jobs.Report) => {
				console.log(report)

				switch (report.type) {
					case 'start':
						break
					case 'paused':
						if (status) status.pause()
						break
					case 'unpaused':
						if (status) status.unpause()
						break
					case 'tile-started':
						if (!status) {
							const textNode = document.createTextNode('UwU')
							await UserInterface.setMessageWorking(textNode)
							status = new JobStatusIndicator(report.tilesTotal, report.pixelsTotal, text => textNode.textContent = text)
						}
						status.reportTileStarted(report.pixelsStarted)
						break
					case 'tile-completed':
						status!.reportTileCompleted(report.pixelsJustCompleted)
						break
					case 'completed':
						break
					case 'aborted':
						break
					case 'errored':
						break
				}
			}
		})

		if (outputBitmap) {
			const downloadLink = document.createElement('a')
			downloadLink.setAttribute('href', 'javascript:void(0)')
			downloadLink.appendChild(document.createTextNode('download'))
			downloadLink.addEventListener('click', async e => {
				e.preventDefault()

				const outputBitmap = UserInterface.getOutputBitmap()
				if (!outputBitmap || UserInterface.getState() !== 'stop') return
				await UserInterface.setState('busy')

				const canvas = new OffscreenCanvas(outputBitmap.width, outputBitmap.height)
				canvas.getContext('bitmaprenderer')!.transferFromImageBitmap(outputBitmap)
				const blob = await canvas.convertToBlob({type: 'image/png'})
				await UserInterface.setOutputBitmap(canvas.transferToImageBitmap())

				try {
					const handle = await showSaveFilePicker({types: [{accept: {'image/png': ['.png']}}], suggestedName: UserInterface.getOutputFilename() + `_waifu2x_${modelSuffix}`})
					const stream = await handle.createWritable({})
					await stream.write(blob)
					await stream.close()
				} catch (e) {
					if (e instanceof DOMException && e.name === 'AbortError') {} else throw e
				} finally {
					await UserInterface.setState('stop')
				}
			})

			await UserInterface.setMessageFace('give', status!.conclude(), '—', downloadLink)
			status = null

			await UserInterface.setOutputBitmap(outputBitmap)
			await UserInterface.setState('stop')
		}
	})

	await UserInterface.setMessageFace('neutral')
})()
