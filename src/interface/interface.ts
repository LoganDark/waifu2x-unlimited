/// <reference path="../createSignal.ts" />
/// <reference path="parameter.ts" />
/// <reference path="message.ts" />

namespace UserInterface {
	const _inputCanvas = documentLoaded.then(() => document.getElementById('input-canvas') as HTMLCanvasElement)
	const _outputCanvas = documentLoaded.then(() => document.getElementById('output-canvas') as HTMLCanvasElement)
	let inputBitmap: ImageBitmap | null = null

	const _form = documentLoaded.then(() => document.getElementById('form') as HTMLFormElement)
	const _parameters = documentLoaded.then(() => ({
		inputFile: new Parameter.FilePicker(document.getElementById('input-file') as HTMLInputElement, setInputImage),
		model: new Parameter.ModelSelect(document.getElementById('select-model') as HTMLSelectElement, 'settings.model', loadSelectedModel),
		noiseLevel: new Parameter.NoiseLevelSelect(document.getElementById('select-noise-level') as HTMLSelectElement, 'settings.noiseLevel', loadSelectedModel),
		enableAntialias: new Parameter.Checkbox(document.getElementById('enable-antialias') as HTMLInputElement, 'settings.enableAntialias'),
		scale: new Parameter.IntegerSelect(document.getElementById('select-scale') as HTMLSelectElement, 'settings.scale', loadSelectedModel),
		tileSize: new Parameter.IntegerInput(document.getElementById('tile-size') as HTMLInputElement, 'settings.tileSize', update),
		tileRandom: new Parameter.Checkbox(document.getElementById('tile-random') as HTMLInputElement, 'settings.tileRandom'),
		tileFocus: new Parameter.Checkbox(document.getElementById('tile-focus') as HTMLInputElement, 'settings.tileFocus'),
		ttaLevel: new Parameter.TtaLevelSelect(document.getElementById('select-tta-level') as HTMLSelectElement, 'settings.ttaLevel'),
		alphaChannel: new Parameter.Checkbox(document.getElementById('alpha-channel') as HTMLInputElement, 'settings.alphaChannel', update),
		backgroundColor: new Parameter.ColorInput(document.getElementById('background-color') as HTMLInputElement, 'settings.backgroundColor'),
		alphaThreshold: new Parameter.NumberInput(document.getElementById('alpha-threshold') as HTMLInputElement, 'settings.alphaThreshold')
	}))

	const _scaleComment = documentLoaded.then(() => document.getElementById('scale-comment') as HTMLSpanElement)
	const _tileSizeComment = documentLoaded.then(() => document.getElementById('tile-size-comment') as HTMLSpanElement)

	const _startButton = documentLoaded.then(() => document.getElementById('start-button') as HTMLButtonElement)
	const _stopButton = documentLoaded.then(() => document.getElementById('stop-button') as HTMLButtonElement)

	export interface InitParameters {
		image: ImageBitmap
		output: HTMLCanvasElement | OffscreenCanvas
		scale: number
		tileSize: number
		alphaChannel: boolean
		backgroundColor: readonly [number, number, number]
		alphaThreshold: number

		model: Model
		antialias: boolean
		ttaLevel: Utils.TtaLevel
	}

	export interface LiveParameters {
		tileRandom(): boolean

		tileFocus(): readonly [number, number] | null
	}

	let state: 'preinit' | Jobs.State | 'busy'
	let [fireStateChanged, onStateChanged] = createSignal<typeof state>()
	state = 'preinit'

	export const getNextState = onStateChanged
	export const getState = () => state
	export const setState = async (newState: typeof state) => {
		state = newState
		fireStateChanged(newState)
		await update()
	}

	let startCallback: ((params: InitParameters, liveParams: LiveParameters) => Promise<void>)

	export const setStartCallback = async (newStartCallback: typeof startCallback) => {
		if (state === 'preinit' || state === 'stop') {
			startCallback = newStartCallback
			await setState('stop')
		}
	}

	let outputBitmap: ImageBitmap | null = null
	let outputFilename: string | null = null

	export const getOutputFilename = () => outputFilename
	export const getOutputBitmap = () => outputBitmap
	export const setOutputBitmap = async (newOutputBitmap: ImageBitmap | null) => {
		outputBitmap = newOutputBitmap
		await update()
	}

	const initParamsTweakable = () => state === 'stop'
	const liveParamsTweakable = () => state === 'stop' || state === 'running' || state === 'paused'

	const update = async () => {
		const outputCanvas = await _outputCanvas
		const parameters = await _parameters
		const scaleComment = await _scaleComment
		const tileSizeComment = await _tileSizeComment
		const startButton = await _startButton
		const stopButton = await _stopButton

		for (const initParam of [
			parameters.inputFile,
			parameters.model,
			parameters.noiseLevel,
			parameters.enableAntialias,
			parameters.scale,
			parameters.tileSize,
			parameters.ttaLevel,
			parameters.alphaChannel,
			parameters.backgroundColor,
			parameters.alphaThreshold
		] as Parameter.Disableable[]) {
			initParam.disabled = !initParamsTweakable()
		}

		parameters.backgroundColor.disabled ||= parameters.alphaChannel.value

		for (const liveParam of [parameters.tileRandom, parameters.tileFocus]) {
			liveParam.disabled = !liveParamsTweakable()
		}

		// snap back to 2x if a non-4x-supporting model is selected
		const [model, style] = parameters.model.value.split('.')
		const modelMetadata = ModelMetadata.models[model]

		// reset scale if too high
		if (modelMetadata?.scales?.has(parameters.scale.value) === false)
			parameters.scale.value = [...modelMetadata.scales.keys()][modelMetadata.scales.size - 1]

		scaleComment.classList.toggle('hidden', modelMetadata?.scales?.has(4) !== false)
		tileSizeComment.classList.toggle('hidden', parameters.tileSize.value >= 256 || modelMetadata?.styles?.get(style)?.prefersLargeTiles !== true)

		startButton.disabled = state !== 'paused' && (!initParamsTweakable() || !inputBitmap)
		startButton.innerText = state === 'paused' ? 'Abort' : 'Start'
		stopButton.disabled = state !== 'running' && state !== 'paused'
		stopButton.innerText = state === 'paused' ? 'Resume' : 'Stop'
		outputCanvas.draggable = !!outputBitmap
	}

	export const setCanvasSize = (canvas: HTMLCanvasElement | OffscreenCanvas, width: number, height: number) => {
		canvas.width = width
		canvas.height = height

		if (canvas instanceof HTMLCanvasElement) {
			canvas.style.setProperty('--canvas-width', `${width / devicePixelRatio}px`)
			canvas.style.setProperty('--canvas-height', `${height / devicePixelRatio}px`)
		}
	}

	const basename = (name: string) => name.replace(/(?:\.\w+)+/g, '')
	const setInputImage = async (newValue: ImageBitmapSource | null) => {
		if (newValue === null) return
		while (!initParamsTweakable()) await onStateChanged()
		const state = getState()
		await setState('busy')
		await setMessageWorking('Loading image')

		const inputCanvas = await _inputCanvas
		inputBitmap = newValue instanceof ImageBitmap ? newValue : await createImageBitmap(newValue, {premultiplyAlpha: 'none'})
		outputFilename = newValue instanceof File ? basename(newValue.name) : null
		setCanvasSize(inputCanvas, inputBitmap.width, inputBitmap.height)
		inputCanvas.getContext('bitmaprenderer')!.transferFromImageBitmap(await createImageBitmap(inputBitmap, {premultiplyAlpha: 'premultiply'}))

		await setState(state)
		await setMessageFace('good')
	}

	const getFileItem = (dt: DataTransfer) => {
		const firstItem = dt.items[0]
		if (!firstItem?.type?.startsWith('image/')) return null
		return firstItem
	}

	const getFile = (dt: DataTransfer) => getFileItem(dt)?.getAsFile() ?? null

	document.addEventListener('paste', async e => {
		if (!initParamsTweakable() || !e.clipboardData) return

		const file = getFile(e.clipboardData)
		if (!file) return

		e.preventDefault()
		const {inputFile} = await _parameters
		inputFile.element.files = null
		await setInputImage(file)
	})

	document.addEventListener('dragover', e => {
		if (initParamsTweakable() && e.dataTransfer && getFileItem(e.dataTransfer)) e.preventDefault()
	})

	document.addEventListener('drop', async e => {
		if (!initParamsTweakable() || !e.dataTransfer) return

		const file = getFile(e.dataTransfer)
		if (!file) return

		e.preventDefault()
		const {inputFile} = await _parameters
		inputFile.element.files = null
		await setInputImage(file)
	})

	Promise.all([_inputCanvas, _outputCanvas]).then(([inputCanvas, outputCanvas]) => {
		let dragging: ImageBitmap | null = null

		outputCanvas.addEventListener('drag', () => dragging = outputBitmap)
		outputCanvas.addEventListener('dragend', () => dragging = null)

		inputCanvas.addEventListener('dragover', e => {
			if (initParamsTweakable() && dragging !== null) e.preventDefault()
		})

		inputCanvas.addEventListener('drop', async e => {
			if (!initParamsTweakable() || dragging === null) return
			e.preventDefault()
			await setOutputBitmap(null) // prevent it from being closed
			await setInputImage(dragging)
		})

		for (const canvas of [inputCanvas, outputCanvas]) {
			canvas.addEventListener('click', () => {
				const classList = canvas.classList

				if (classList.contains('expanded2')) {
					classList.remove('expanded2')
				} else if (classList.contains('expanded')) {
					classList.replace('expanded', 'expanded2')
				} else {
					classList.add('expanded')
				}
			})
		}
	})

	const getSelectedModel = async () => {
		const parameters = await _parameters
		const [name, style] = parameters.model.value.split('.')
		return await Model.Cache.load(name, style, parameters.scale.value, parameters.noiseLevel.value)
	}

	const loadSelectedModel = async () => {
		while (!initParamsTweakable()) await onStateChanged()
		const state = getState()
		await setState('busy')
		await setMessageWorking('Loading model')
		await getSelectedModel()
		await setState(state)
		await setMessageFace('good')
	}

	Promise.all([_form, _outputCanvas, _startButton, _stopButton]).then(([form, outputCanvas, startButton, stopButton]) => {
		let mouseFocus: ReturnType<LiveParameters['tileFocus']> = null

		const mouseHandler = (e: MouseEvent) => {
			const rect = outputCanvas.getBoundingClientRect()
			mouseFocus = [(e.clientX - rect.x) * (outputCanvas.width / rect.width), (e.clientY - rect.y) * (outputCanvas.height / rect.height)] as const
		}

		outputCanvas.addEventListener('mouseenter', mouseHandler)
		outputCanvas.addEventListener('mousemove', mouseHandler)
		outputCanvas.addEventListener('mouseleave', () => mouseFocus = null)

		form.addEventListener('submit', async e => {
			e.preventDefault()
			if (!initParamsTweakable() || !inputBitmap) return

			outputBitmap?.close()
			outputBitmap = null

			const params = await _parameters
			const backgroundColor = params.backgroundColor.value
			const initParams: InitParameters = {
				image: inputBitmap,
				output: await _outputCanvas,
				scale: params.scale.value,
				tileSize: params.tileSize.value,
				alphaChannel: params.alphaChannel.value,
				backgroundColor: [(backgroundColor >> 16) / 255, (backgroundColor >> 8 & 0xFF) / 255, (backgroundColor & 0xFF) / 255],
				alphaThreshold: params.alphaThreshold.value,
				model: await getSelectedModel(),
				antialias: params.enableAntialias.value,
				ttaLevel: params.ttaLevel.value
			}

			const liveParams: LiveParameters = {
				tileRandom: () => params.tileRandom.value,
				tileFocus: () => params.tileFocus.value ? mouseFocus : null
			}

			if (state === 'paused') await setState('stop')
			await setState('running')
			await setMessageWorking('Starting render job')
			await setCanvasSize(initParams.output, inputBitmap.width * initParams.scale, inputBitmap.height * initParams.scale)

			let exitedCleanly = false

			try {
				await startCallback(initParams, liveParams)
				exitedCleanly = true
			} finally {
				if (!exitedCleanly) await setState('stop')
			}
		})

		startButton.addEventListener('click', async e => {
			if (state === 'paused') {
				e.preventDefault()
				await setState('stop')
				await setMessageFace('good', 'Render job aborted')
			}
		})

		stopButton.addEventListener('click', async () => {
			if (state === 'running') {
				await setState('paused')
				await setMessageFace('good', 'Render job paused')
			} else if (state === 'paused') {
				await setState('running')
			}
		})
	})
}
