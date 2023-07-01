"use strict";
const createSignal = () => {
    let fire, next = (function refresh() {
        return new Promise(resolve => fire = (value) => {
            next = refresh();
            resolve(value);
        });
    })();
    return [(value) => fire(value), () => next];
};
const documentLoaded = document.readyState != 'loading'
    ? Promise.resolve()
    : new Promise(resolve => document.addEventListener('DOMContentLoaded', () => resolve()));
var ModelMetadata;
(function (ModelMetadata) {
    ModelMetadata.models = {
        swin_unet: {
            styles: new Map([
                ['art', { prefersLargeTiles: false }],
                ['art_scan', { prefersLargeTiles: true }],
                ['photo', { prefersLargeTiles: true }]
            ]),
            scales: new Map([
                [1, { context: 10 }],
                [2, { context: 16 }],
                [4, { context: 12 }]
            ])
        },
        cunet: {
            styles: new Map([
                ['art', { prefersLargeTiles: false }]
            ]),
            scales: new Map([
                [1, { context: 28 }],
                [2, { context: 18 }]
            ])
        }
    };
    ModelMetadata.pathBase = 'models/';
    ModelMetadata.getModelBasename = (scale, noise) => (noise === -1 ? `scale${scale}x` : scale === 1 ? `noise${noise}` : `noise${noise}_scale${scale}x`);
    ModelMetadata.getModelPath = (name, style, scale, noise, suffix = '') => `${ModelMetadata.pathBase}${name}/${style}/${ModelMetadata.getModelBasename(scale, noise)}${suffix}.onnx`;
    ModelMetadata.getUtilityPath = (name) => `${ModelMetadata.pathBase}utils/${name}.onnx`;
})(ModelMetadata || (ModelMetadata = {}));
/// <reference path="metadata.ts" />
class Session {
    session;
    constructor(session) {
        this.session = session;
    }
    _run = async (feeds, outputName) => (await this.session.run(feeds))[outputName];
}
class Utils {
    _pad;
    _tta;
    _antialias;
    pad;
    pad_alpha;
    tta_split;
    tta_merge;
    antialias;
    constructor(_pad, _tta, _antialias, pad = _pad.pad, pad_alpha = _pad.pad_alpha, tta_split = _tta.split, tta_merge = _tta.merge, antialias = _antialias.antialias) {
        this._pad = _pad;
        this._tta = _tta;
        this._antialias = _antialias;
        this.pad = pad;
        this.pad_alpha = pad_alpha;
        this.tta_split = tta_split;
        this.tta_merge = tta_merge;
        this.antialias = antialias;
    }
    static load = async () => new this(await Utils.Padding.load(), await Utils.TTA.load(), await Utils.Antialias.load());
    static sessionOptions = {
        executionProviders: ['wasm'],
        executionMode: 'parallel',
        graphOptimizationLevel: 'all'
    };
    static createSession = (path) => ort.InferenceSession.create(path, this.sessionOptions);
    static createUtilitySession = (name) => this.createSession(ModelMetadata.getUtilityPath(name));
    static createModelSession = (name, style, scale, noise, suffix) => this.createSession(ModelMetadata.getModelPath(name, style, scale, noise, suffix));
}
(function (Utils) {
    class Padding {
        _pad;
        _alpha;
        pad;
        pad_alpha;
        constructor(_pad, _alpha, pad = _pad.pad, pad_alpha = _alpha.pad) {
            this._pad = _pad;
            this._alpha = _alpha;
            this.pad = pad;
            this.pad_alpha = pad_alpha;
        }
        static load = async () => new this(await Padding.Pad.load(), await Padding.AlphaPad.load());
    }
    Utils.Padding = Padding;
    (function (Padding) {
        class Pad extends Session {
            static load = async () => new this(await Utils.createUtilitySession('pad'));
            pad = async (image, padding) => await this._run({
                x: image,
                left: new ort.Tensor('int64', BigInt64Array.from([BigInt(padding.left)]), []),
                right: new ort.Tensor('int64', BigInt64Array.from([BigInt(padding.right)]), []),
                top: new ort.Tensor('int64', BigInt64Array.from([BigInt(padding.top)]), []),
                bottom: new ort.Tensor('int64', BigInt64Array.from([BigInt(padding.bottom)]), [])
            }, 'y');
        }
        Padding.Pad = Pad;
        class AlphaPad extends Session {
            static load = async () => new this(await Utils.createUtilitySession('alpha_border_padding'));
            pad = async (rgb, alpha, offset) => {
                const output = await this._run({
                    rgb: new ort.Tensor('float32', rgb.data, rgb.dims.slice(1)),
                    alpha: new ort.Tensor('float32', alpha.data, alpha.dims.slice(1)),
                    offset: new ort.Tensor('int64', BigInt64Array.from([BigInt(offset)]), [])
                }, 'y');
                return new ort.Tensor('float32', output.data, [1, ...output.dims]);
            };
        }
        Padding.AlphaPad = AlphaPad;
    })(Padding = Utils.Padding || (Utils.Padding = {}));
    class TTA {
        _split;
        _merge;
        split;
        merge;
        constructor(_split, _merge, split = _split.split, merge = _merge.merge) {
            this._split = _split;
            this._merge = _merge;
            this.split = split;
            this.merge = merge;
        }
        static load = async () => new this(await TTA.Split.load(), await TTA.Merge.load());
    }
    Utils.TTA = TTA;
    (function (TTA) {
        class Split extends Session {
            static load = async () => new this(await Utils.createUtilitySession('tta_split'));
            split = async (image, level) => await this._run({
                x: image,
                tta_level: new ort.Tensor('int64', BigInt64Array.from([BigInt(level)]), [])
            }, 'y');
        }
        TTA.Split = Split;
        class Merge extends Session {
            static load = async () => new this(await Utils.createUtilitySession('tta_merge'));
            merge = async (image, level) => await this._run({
                x: image,
                tta_level: new ort.Tensor('int64', BigInt64Array.from([BigInt(level)]), [])
            }, 'y');
        }
        TTA.Merge = Merge;
    })(TTA = Utils.TTA || (Utils.TTA = {}));
    class Antialias extends Session {
        static load = async () => new this(await Utils.createUtilitySession('antialias'));
        antialias = async (image) => await this._run({ x: image }, 'y');
    }
    Utils.Antialias = Antialias;
})(Utils || (Utils = {}));
class Model extends Session {
    _name;
    _style;
    _scale;
    _noise;
    _suffix;
    _context;
    constructor(session, _name, _style, _scale, _noise, _suffix, _context) {
        super(session);
        this._name = _name;
        this._style = _style;
        this._scale = _scale;
        this._noise = _noise;
        this._suffix = _suffix;
        this._context = _context;
    }
    get name() { return this._name; }
    get style() { return this._style; }
    get scale() { return this._scale; }
    get noise() { return this._noise; }
    get suffix() { return this._suffix; }
    get context() { return this._context; }
    static load = async (name, style, scale, noise, suffix) => new this(await Utils.createModelSession(name, style, scale, noise, suffix), name, style, scale, noise, suffix ?? null, ModelMetadata.models[name].scales.get(scale).context);
    run = async (image) => await this._run({ x: image }, 'y');
}
(function (Model) {
    var Cache;
    (function (Cache) {
        const cache = {};
        Cache.load = (name, style, scale, noise, suffix) => {
            const key = `${name}.${style}:${suffix}:${scale}:${noise}}`;
            return cache[key] ?? (cache[key] = Model.load(name, style, scale, noise, suffix));
        };
    })(Cache = Model.Cache || (Model.Cache = {}));
})(Model || (Model = {}));
/// <reference path="../metadata.ts" />
var UserInterface;
(function (UserInterface) {
    class Parameter {
        element;
        _onValueChanged;
        _defaultValue;
        constructor(element, _onValueChanged = () => { }) {
            this.element = element;
            this._onValueChanged = _onValueChanged;
            [this._fireValueChanged, this.getNextValue] = createSignal();
            this._defaultValue = this._getValue();
        }
        _init() {
            let savedValue = null;
            try {
                savedValue = this._getSavedValue();
            }
            catch { }
            if (savedValue !== null && savedValue !== this._getValue())
                this._setValue(savedValue);
            const value = savedValue === null ? this._getValue() : savedValue;
            this._setSavedValue(value);
            this._onValueChanged(value);
        }
        _fireValueChanged;
        getNextValue;
        get value() { return this._getValue(); }
        set value(newValue) {
            if (this._setValue(newValue)) {
                this._setSavedValue(newValue);
                //this._onValueChanged(newValue)
                this._fireValueChanged(newValue);
            }
        }
        update() {
            const newValue = this.value;
            this._setSavedValue(newValue);
            this._onValueChanged(newValue);
            this._fireValueChanged(newValue);
        }
    }
    UserInterface.Parameter = Parameter;
    (function (Parameter) {
        class FilePicker extends Parameter {
            constructor(picker, _onValueChanged) {
                super(picker, _onValueChanged);
                picker.addEventListener('change', () => this.update());
                this._init();
            }
            _getSavedValue() { return null; }
            _setSavedValue(newValue) { }
            _getValue() { return this.element.files?.item(0) ?? null; }
            _setValue(newValue) { return newValue == this._getValue(); }
            get disabled() { return this.element.disabled; }
            set disabled(disabled) { this.element.disabled = disabled; }
        }
        Parameter.FilePicker = FilePicker;
        class Select extends Parameter {
            key;
            constructor(select, key, _onValueChanged) {
                super(select, _onValueChanged);
                this.key = key;
                select.addEventListener('change', () => this.update());
                this._init();
            }
            _getSavedValue() {
                let stored = null;
                try {
                    stored = localStorage.getItem(this.key);
                }
                catch { }
                return stored !== null ? this._fromString(stored) : null;
            }
            _setSavedValue(newValue) {
                const toStore = this._toString(newValue);
                try {
                    localStorage.setItem(this.key, toStore);
                }
                catch { }
            }
            _getValue() {
                const value = this._fromString(this.element.value);
                if (value === null) {
                    throw new TypeError('current element value is invalid');
                }
                else {
                    return value;
                }
            }
            _setValue(newValue) {
                const value = this._toString(newValue);
                if (value !== this.element.value) {
                    this.element.value = value;
                    return true;
                }
                else {
                    return false;
                }
            }
            get disabled() { return this.element.disabled; }
            set disabled(disabled) { this.element.disabled = disabled; }
        }
        Parameter.Select = Select;
        class Checkbox extends Parameter {
            key;
            constructor(checkbox, key, _onValueChanged) {
                if (checkbox.type !== 'checkbox')
                    throw new TypeError('expected checkbox');
                super(checkbox, _onValueChanged);
                this.key = key;
                checkbox.addEventListener('change', () => this.update());
                this._init();
            }
            _getSavedValue() {
                try {
                    const stored = localStorage.getItem(this.key);
                    return stored === 'true' ? true : stored === 'false' ? false : null;
                }
                catch { }
                return null;
            }
            _setSavedValue(newValue) {
                try {
                    localStorage.setItem(this.key, newValue ? 'true' : 'false');
                }
                catch { }
            }
            _getValue() {
                return this.element.checked;
            }
            _setValue(newValue) {
                if (newValue !== this.element.checked) {
                    this.element.checked = newValue;
                    return true;
                }
                else {
                    return false;
                }
            }
            get disabled() { return this.element.disabled; }
            set disabled(disabled) { this.element.disabled = disabled; }
        }
        Parameter.Checkbox = Checkbox;
        class IntegerInput extends Parameter {
            key;
            constructor(input, key, _onValueChanged) {
                if (input.type !== 'number')
                    throw new TypeError('expected number input');
                super(input, _onValueChanged);
                this.key = key;
                input.addEventListener('change', () => this.update());
                this._init();
            }
            _getSavedValue() {
                const stored = localStorage.getItem(this.key);
                const value = Number(stored ?? '!');
                if (!Number.isInteger(value))
                    return null;
                return value;
            }
            _setSavedValue(newValue) {
                try {
                    localStorage.setItem(this.key, newValue.toString());
                }
                catch { }
            }
            _getValue() {
                const value = Number(this.element.value);
                if (!Number.isInteger(value)) {
                    this._setValue(this._defaultValue);
                    return this._defaultValue;
                }
                return value;
            }
            _setValue(newValue) {
                const toSet = newValue.toString();
                if (toSet !== this.element.value) {
                    this.element.value = toSet;
                    return true;
                }
                else {
                    return false;
                }
            }
            get disabled() { return this.element.disabled; }
            set disabled(disabled) { this.element.disabled = disabled; }
        }
        Parameter.IntegerInput = IntegerInput;
        class NumberInput extends Parameter {
            key;
            constructor(input, key, _onValueChanged) {
                if (input.type !== 'number')
                    throw new TypeError('expected number input');
                super(input, _onValueChanged);
                this.key = key;
                input.addEventListener('change', () => this.update());
                this._init();
            }
            _getSavedValue() {
                const stored = localStorage.getItem(this.key);
                const value = Number(stored ?? '!');
                if (!Number.isFinite(value))
                    return null;
                return value;
            }
            _setSavedValue(newValue) {
                try {
                    localStorage.setItem(this.key, newValue.toString());
                }
                catch { }
            }
            _getValue() {
                const value = Number(this.element.value);
                if (!Number.isFinite(value)) {
                    this._setValue(this._defaultValue);
                    return this._defaultValue;
                }
                return value;
            }
            _setValue(newValue) {
                const toSet = newValue.toString();
                if (toSet !== this.element.value) {
                    this.element.value = toSet;
                    return true;
                }
                else {
                    return false;
                }
            }
            get disabled() { return this.element.disabled; }
            set disabled(disabled) { this.element.disabled = disabled; }
        }
        Parameter.NumberInput = NumberInput;
        class ModelSelect extends Select {
            _fromString(value) {
                const [model, style] = value.split('.');
                if (ModelMetadata.models[model]?.styles?.has(style))
                    return value;
                return null;
            }
            _toString(value) { return value; }
        }
        Parameter.ModelSelect = ModelSelect;
        class IntegerSelect extends Select {
            _fromString(value) {
                const parsed = parseInt(value);
                return parsed !== parsed ? null : parsed;
            }
            _toString(value) {
                return value.toString();
            }
        }
        Parameter.IntegerSelect = IntegerSelect;
        class NoiseLevelSelect extends Select {
            _fromString(value) {
                return ['-1', '0', '1', '2', '3'].indexOf(value) > -1 ? +value : null;
            }
            _toString(value) {
                return value.toString();
            }
        }
        Parameter.NoiseLevelSelect = NoiseLevelSelect;
        class TtaLevelSelect extends Select {
            _fromString(value) {
                return ['0', '2', '4'].indexOf(value) > -1 ? +value : null;
            }
            _toString(value) {
                return value.toString();
            }
        }
        Parameter.TtaLevelSelect = TtaLevelSelect;
        class ColorInput extends Parameter {
            key;
            constructor(input, key, _onValueChanged) {
                if (input.type !== 'color')
                    throw new TypeError('expected color input');
                super(input, _onValueChanged);
                this.key = key;
                input.addEventListener('change', () => this.update());
                this._init();
            }
            _getSavedValue() {
                const stored = localStorage.getItem(this.key);
                const value = parseInt(stored?.match(/^#([0-9a-f]{6})$/i)?.[1] ?? '!', 16);
                if (!Number.isFinite(value))
                    return null;
                return value;
            }
            _setSavedValue(newValue) {
                try {
                    localStorage.setItem(this.key, `#${newValue.toString(16).padStart(6, '0')}`);
                }
                catch { }
            }
            _getValue() {
                const value = parseInt(this.element.value.match(/^#([0-9a-f]{6})$/i)?.[1] ?? '!', 16);
                if (!Number.isFinite(value)) {
                    this._setValue(this._defaultValue);
                    return this._defaultValue;
                }
                return value;
            }
            _setValue(newValue) {
                const toSet = `#${newValue.toString(16).padStart(6, '0')}`;
                if (toSet.toLowerCase() !== this.element.value.toLowerCase()) {
                    this.element.value = toSet;
                    return true;
                }
                else {
                    return false;
                }
            }
            get disabled() { return this.element.disabled; }
            set disabled(disabled) { this.element.disabled = disabled; }
        }
        Parameter.ColorInput = ColorInput;
    })(Parameter = UserInterface.Parameter || (UserInterface.Parameter = {}));
})(UserInterface || (UserInterface = {}));
/// <reference path="../documentLoaded.ts" />
/// <reference path="../createSignal.ts" />
var UserInterface;
(function (UserInterface) {
    const _message = documentLoaded.then(() => document.getElementById('message'));
    let [onMessageChanged, messageChanged] = createSignal();
    UserInterface.setMessage = async (...contents) => {
        const message = await _message;
        message.replaceChildren(...contents);
        onMessageChanged();
    };
    const faces = {
        neutral: '( ・∀・)',
        happy: '(ﾟ∀ﾟ)',
        good: '( ・∀・)b',
        give: '( ・∀・)つ',
        shock: '(・A・)',
        panic: '(・A・)!!',
        unamused: '(-_-)'
    };
    UserInterface.setMessageFace = (face, ...contents) => UserInterface.setMessage(`${faces[face]}${contents.length > 0 ? '　' : ''}`, ...contents);
    UserInterface.setMessageWorking = async (...contents) => {
        await _message;
        let currentFace = '( ・∀・)φ　 ', nextFace = '( ・∀・) φ　';
        const faceAnimationNode = document.createTextNode(currentFace);
        const faceInterval = setInterval(() => {
            const toSet = nextFace;
            faceAnimationNode.textContent = toSet;
            nextFace = currentFace;
            currentFace = toSet;
        }, 500);
        let dotPos = 0;
        const numDots = 3;
        const dotsAnimationNode = document.createTextNode(` ${'.'.repeat(numDots)}`);
        const dotsInterval = setInterval(() => {
            dotPos += 1;
            dotPos %= numDots + 1;
            dotsAnimationNode.textContent = `${'.'.repeat(dotPos)} ${'.'.repeat(numDots - dotPos)}`;
        }, 250);
        await UserInterface.setMessage(faceAnimationNode, ...contents, dotsAnimationNode);
        messageChanged().then(() => clearInterval(faceInterval));
        messageChanged().then(() => clearInterval(dotsInterval));
    };
    window.addEventListener('unhandledrejection', async (e) => {
        await UserInterface.setMessageFace('unamused', 'Unexpected error: ', '' + e.reason);
    });
})(UserInterface || (UserInterface = {}));
/// <reference path="../createSignal.ts" />
/// <reference path="parameter.ts" />
/// <reference path="message.ts" />
var UserInterface;
(function (UserInterface) {
    const _inputCanvas = documentLoaded.then(() => document.getElementById('input-canvas'));
    const _outputCanvas = documentLoaded.then(() => document.getElementById('output-canvas'));
    let inputBitmap = null;
    const _form = documentLoaded.then(() => document.getElementById('form'));
    const _parameters = documentLoaded.then(() => ({
        inputFile: new UserInterface.Parameter.FilePicker(document.getElementById('input-file'), setInputImage),
        model: new UserInterface.Parameter.ModelSelect(document.getElementById('select-model'), 'settings.model', loadSelectedModel),
        noiseLevel: new UserInterface.Parameter.NoiseLevelSelect(document.getElementById('select-noise-level'), 'settings.noiseLevel', loadSelectedModel),
        enableAntialias: new UserInterface.Parameter.Checkbox(document.getElementById('enable-antialias'), 'settings.enableAntialias'),
        scale: new UserInterface.Parameter.IntegerSelect(document.getElementById('select-scale'), 'settings.scale', loadSelectedModel),
        tileSize: new UserInterface.Parameter.IntegerInput(document.getElementById('tile-size'), 'settings.tileSize', update),
        tileRandom: new UserInterface.Parameter.Checkbox(document.getElementById('tile-random'), 'settings.tileRandom'),
        tileFocus: new UserInterface.Parameter.Checkbox(document.getElementById('tile-focus'), 'settings.tileFocus'),
        ttaLevel: new UserInterface.Parameter.TtaLevelSelect(document.getElementById('select-tta-level'), 'settings.ttaLevel'),
        alphaChannel: new UserInterface.Parameter.Checkbox(document.getElementById('alpha-channel'), 'settings.alphaChannel', update),
        backgroundColor: new UserInterface.Parameter.ColorInput(document.getElementById('background-color'), 'settings.backgroundColor'),
        alphaThreshold: new UserInterface.Parameter.NumberInput(document.getElementById('alpha-threshold'), 'settings.alphaThreshold')
    }));
    const _scaleComment = documentLoaded.then(() => document.getElementById('scale-comment'));
    const _tileSizeComment = documentLoaded.then(() => document.getElementById('tile-size-comment'));
    const _startButton = documentLoaded.then(() => document.getElementById('start-button'));
    const _stopButton = documentLoaded.then(() => document.getElementById('stop-button'));
    let state;
    let [fireStateChanged, onStateChanged] = createSignal();
    state = 'preinit';
    UserInterface.getNextState = onStateChanged;
    UserInterface.getState = () => state;
    UserInterface.setState = async (newState) => {
        state = newState;
        fireStateChanged(newState);
        await update();
    };
    let startCallback;
    UserInterface.setStartCallback = async (newStartCallback) => {
        if (state === 'preinit' || state === 'stop') {
            startCallback = newStartCallback;
            await UserInterface.setState('stop');
        }
    };
    let outputBitmap = null;
    let outputFilename = null;
    UserInterface.getOutputFilename = () => outputFilename;
    UserInterface.getOutputBitmap = () => outputBitmap;
    UserInterface.setOutputBitmap = async (newOutputBitmap) => {
        outputBitmap = newOutputBitmap;
        await update();
    };
    const initParamsTweakable = () => state === 'stop';
    const liveParamsTweakable = () => state === 'stop' || state === 'running' || state === 'paused';
    const update = async () => {
        const outputCanvas = await _outputCanvas;
        const parameters = await _parameters;
        const scaleComment = await _scaleComment;
        const tileSizeComment = await _tileSizeComment;
        const startButton = await _startButton;
        const stopButton = await _stopButton;
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
        ]) {
            initParam.disabled = !initParamsTweakable();
        }
        parameters.backgroundColor.disabled ||= parameters.alphaChannel.value;
        for (const liveParam of [parameters.tileRandom, parameters.tileFocus]) {
            liveParam.disabled = !liveParamsTweakable();
        }
        // snap back to 2x if a non-4x-supporting model is selected
        const [model, style] = parameters.model.value.split('.');
        const modelMetadata = ModelMetadata.models[model];
        // reset scale if too high
        if (modelMetadata?.scales?.has(parameters.scale.value) === false)
            parameters.scale.value = [...modelMetadata.scales.keys()][modelMetadata.scales.size - 1];
        scaleComment.classList.toggle('hidden', modelMetadata?.scales?.has(4) !== false);
        tileSizeComment.classList.toggle('hidden', parameters.tileSize.value >= 256 || modelMetadata?.styles?.get(style)?.prefersLargeTiles !== true);
        startButton.disabled = state !== 'paused' && (!initParamsTweakable() || !inputBitmap);
        startButton.innerText = state === 'paused' ? 'Abort' : 'Start';
        stopButton.disabled = state !== 'running' && state !== 'paused';
        stopButton.innerText = state === 'paused' ? 'Resume' : 'Stop';
        outputCanvas.draggable = !!outputBitmap;
    };
    UserInterface.setCanvasSize = (canvas, width, height) => {
        canvas.width = width;
        canvas.height = height;
        if (canvas instanceof HTMLCanvasElement) {
            canvas.style.setProperty('--canvas-width', `${width / devicePixelRatio}px`);
            canvas.style.setProperty('--canvas-height', `${height / devicePixelRatio}px`);
        }
    };
    const basename = (name) => name.replace(/(?:\.\w+)+/g, '');
    const setInputImage = async (newValue) => {
        if (newValue === null)
            return;
        while (!initParamsTweakable())
            await onStateChanged();
        const state = UserInterface.getState();
        await UserInterface.setState('busy');
        await UserInterface.setMessageWorking('Loading image');
        const inputCanvas = await _inputCanvas;
        inputBitmap = newValue instanceof ImageBitmap ? newValue : await createImageBitmap(newValue, { premultiplyAlpha: 'none' });
        outputFilename = newValue instanceof File ? basename(newValue.name) : null;
        UserInterface.setCanvasSize(inputCanvas, inputBitmap.width, inputBitmap.height);
        inputCanvas.getContext('bitmaprenderer').transferFromImageBitmap(await createImageBitmap(inputBitmap, { premultiplyAlpha: 'premultiply' }));
        await UserInterface.setState(state);
        await UserInterface.setMessageFace('good');
    };
    const getFileItem = (dt) => {
        const firstItem = dt.items[0];
        if (!firstItem?.type?.startsWith('image/'))
            return null;
        return firstItem;
    };
    const getFile = (dt) => getFileItem(dt)?.getAsFile() ?? null;
    document.addEventListener('paste', async (e) => {
        if (!initParamsTweakable() || !e.clipboardData)
            return;
        const file = getFile(e.clipboardData);
        if (!file)
            return;
        e.preventDefault();
        const { inputFile } = await _parameters;
        inputFile.element.files = null;
        await setInputImage(file);
    });
    document.addEventListener('dragover', e => {
        if (initParamsTweakable() && e.dataTransfer && getFileItem(e.dataTransfer))
            e.preventDefault();
    });
    document.addEventListener('drop', async (e) => {
        if (!initParamsTweakable() || !e.dataTransfer)
            return;
        const file = getFile(e.dataTransfer);
        if (!file)
            return;
        e.preventDefault();
        const { inputFile } = await _parameters;
        inputFile.element.files = null;
        await setInputImage(file);
    });
    Promise.all([_inputCanvas, _outputCanvas]).then(([inputCanvas, outputCanvas]) => {
        let dragging = null;
        outputCanvas.addEventListener('drag', () => dragging = outputBitmap);
        outputCanvas.addEventListener('dragend', () => dragging = null);
        inputCanvas.addEventListener('dragover', e => {
            if (initParamsTweakable() && dragging !== null)
                e.preventDefault();
        });
        inputCanvas.addEventListener('drop', async (e) => {
            if (!initParamsTweakable() || dragging === null)
                return;
            e.preventDefault();
            await UserInterface.setOutputBitmap(null); // prevent it from being closed
            await setInputImage(dragging);
        });
        for (const canvas of [inputCanvas, outputCanvas]) {
            canvas.addEventListener('click', () => {
                const classList = canvas.classList;
                if (classList.contains('expanded2')) {
                    classList.remove('expanded2');
                }
                else if (classList.contains('expanded')) {
                    classList.replace('expanded', 'expanded2');
                }
                else {
                    classList.add('expanded');
                }
            });
        }
    });
    const getSelectedModel = async () => {
        const parameters = await _parameters;
        const [name, style] = parameters.model.value.split('.');
        return await Model.Cache.load(name, style, parameters.scale.value, parameters.noiseLevel.value);
    };
    const loadSelectedModel = async () => {
        while (!initParamsTweakable())
            await onStateChanged();
        const state = UserInterface.getState();
        await UserInterface.setState('busy');
        await UserInterface.setMessageWorking('Loading model');
        await getSelectedModel();
        await UserInterface.setState(state);
        await UserInterface.setMessageFace('good');
    };
    Promise.all([_form, _outputCanvas, _startButton, _stopButton]).then(([form, outputCanvas, startButton, stopButton]) => {
        let mouseFocus = null;
        const mouseHandler = (e) => {
            const rect = outputCanvas.getBoundingClientRect();
            mouseFocus = [(e.clientX - rect.x) * (outputCanvas.width / rect.width), (e.clientY - rect.y) * (outputCanvas.height / rect.height)];
        };
        outputCanvas.addEventListener('mouseenter', mouseHandler);
        outputCanvas.addEventListener('mousemove', mouseHandler);
        outputCanvas.addEventListener('mouseleave', () => mouseFocus = null);
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!initParamsTweakable() || !inputBitmap)
                return;
            outputBitmap?.close();
            outputBitmap = null;
            const params = await _parameters;
            const backgroundColor = params.backgroundColor.value;
            const initParams = {
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
            };
            const liveParams = {
                tileRandom: () => params.tileRandom.value,
                tileFocus: () => params.tileFocus.value ? mouseFocus : null
            };
            if (state === 'paused')
                await UserInterface.setState('stop');
            await UserInterface.setState('running');
            await UserInterface.setMessageWorking('Starting render job');
            await UserInterface.setCanvasSize(initParams.output, inputBitmap.width * initParams.scale, inputBitmap.height * initParams.scale);
            let exitedCleanly = false;
            try {
                await startCallback(initParams, liveParams);
                exitedCleanly = true;
            }
            finally {
                if (!exitedCleanly)
                    await UserInterface.setState('stop');
            }
        });
        startButton.addEventListener('click', async (e) => {
            if (state === 'paused') {
                e.preventDefault();
                await UserInterface.setState('stop');
                await UserInterface.setMessageFace('good', 'Render job aborted');
            }
        });
        stopButton.addEventListener('click', async () => {
            if (state === 'running') {
                await UserInterface.setState('paused');
                await UserInterface.setMessageFace('good', 'Render job paused');
            }
            else if (state === 'paused') {
                await UserInterface.setState('running');
            }
        });
    });
})(UserInterface || (UserInterface = {}));
class JobStatusIndicator {
    _totalTiles;
    _totalPixels;
    _callback;
    _start;
    _raf;
    _completedTiles = 0;
    _completedPixels = 0;
    _tileLastCompleted = null;
    _largestTilePixels = 0;
    _currentTilePixels = 0;
    _pausedAt = null;
    constructor(_totalTiles, _totalPixels, _callback) {
        this._totalTiles = _totalTiles;
        this._totalPixels = _totalPixels;
        this._callback = _callback;
        this._start = performance.now();
        this._raf = requestAnimationFrame(this.tick);
    }
    tick = (now) => {
        try {
            const since = now - this._start;
            // precision so that every tile completed results in a percentage change
            const precision = Math.max(Math.log10(this._totalTiles / 100), 0);
            let message = `${this._completedTiles}/${this._totalTiles} (${(this._completedPixels / this._totalPixels * 100).toFixed(precision)}% complete)`;
            if (this._tileLastCompleted !== null) {
                const forCurrentPixels = this._tileLastCompleted - this._start;
                const currentLargestTiles = this._completedPixels / this._largestTilePixels;
                const perLargestTile = forCurrentPixels / currentLargestTiles;
                message += ` (${perLargestTile.toFixed(1)}ms/t`;
                if (perLargestTile > 1500) {
                    if (this._completedTiles < this._totalTiles) {
                        const perCurrentTile = this._currentTilePixels / this._largestTilePixels * perLargestTile;
                        const estimatedProgress = Math.min((since - forCurrentPixels) / perCurrentTile * 0.95, 1);
                        // precision so that every 100ms results in a percentage change
                        const precision = Math.max(Math.log10(perLargestTile / 1000), 0);
                        message += `; ${(estimatedProgress * 100).toFixed(precision)}%${estimatedProgress === 1 ? '?' : ''})`;
                    }
                    else {
                        message += ')';
                    }
                }
                else {
                    message += `; ${(1000 / perLargestTile).toFixed(2)}t/s)`;
                }
            }
            this._callback(message);
        }
        finally {
            this._raf = requestAnimationFrame(this.tick);
        }
    };
    pause() {
        if (this._raf === null)
            throw new TypeError('already paused');
        cancelAnimationFrame(this._raf);
        this._pausedAt = performance.now();
    }
    unpause() {
        if (this._pausedAt === null)
            throw new TypeError('not paused');
        const now = performance.now();
        const duration = now - this._pausedAt;
        this._start += duration;
        if (this._tileLastCompleted !== null)
            this._tileLastCompleted += duration;
        this._pausedAt = null;
        this._raf = requestAnimationFrame(this.tick);
    }
    reportTileStarted(pixels) {
        this._currentTilePixels = pixels;
    }
    reportTileCompleted(pixels) {
        this._tileLastCompleted = performance.now();
        this._completedTiles++;
        this._completedPixels += pixels;
        if (pixels > this._largestTilePixels)
            this._largestTilePixels = pixels;
    }
    conclude() {
        this.pause();
        const duration = this._pausedAt - this._start;
        return `${duration.toFixed(2)}ms`;
    }
}
var Convert;
(function (Convert) {
    class GlBlitter {
        width;
        height;
        gl;
        constructor(width, height) {
            this.width = width;
            this.height = height;
            this.gl = new OffscreenCanvas(0, 0).getContext('webgl');
            if (!this.gl)
                throw new TypeError('failed to acquire WebGL rendering context');
            this.gl.activeTexture(this.gl.TEXTURE0);
            const texture = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.gl.createFramebuffer());
            this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, texture, 0);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, width, height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
        }
        writePixels(source, x, y) {
            this.gl.texSubImage2D(this.gl.TEXTURE_2D, 0, x, y, this.gl.RGBA, this.gl.UNSIGNED_BYTE, source);
        }
        toImageData() {
            const imageData = new ImageData(this.width, this.height);
            this.gl.readPixels(0, 0, this.width, this.height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, imageData.data);
            return imageData;
        }
    }
    Convert.GlBlitter = GlBlitter;
    const gl = new OffscreenCanvas(0, 0).getContext('webgl');
    {
        gl.activeTexture(gl.TEXTURE0);
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.bindFramebuffer(gl.FRAMEBUFFER, gl.createFramebuffer());
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    }
    Convert.readPixels = (source, dest) => {
        const { width, height } = source;
        if (dest.length !== width * height * 4)
            throw new TypeError('invalid destination length');
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
        if (source instanceof ImageBitmap)
            source.close();
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, dest);
    };
    Convert.toRgbAlpha = ({ width, height, data }) => {
        const pixels = width * height;
        const rgb = new Float32Array(pixels * 3);
        const alpha = new Float32Array(pixels);
        let r = 0, g = pixels, b = pixels * 2;
        for (let i = 0; i < data.length; i += 4)
            rgb[r++] = data[i] / 255;
        for (let i = 1; i < data.length; i += 4)
            rgb[g++] = data[i] / 255;
        for (let i = 2; i < data.length; i += 4)
            rgb[b++] = data[i] / 255;
        for (let i = 3, a = 0; i < data.length; i += 4)
            alpha[a++] = data[i] / 255;
        return [
            new ort.Tensor('float32', rgb, [1, 3, height, width]),
            new ort.Tensor('float32', alpha, [1, 1, height, width])
        ];
    };
    Convert.stretchAlpha = ({ dims, data: alpha1 }) => {
        const [height, width] = dims.slice(2);
        const pixels = width * height;
        const alpha3 = new Float32Array(pixels * 3);
        alpha3.set(alpha1, 0);
        alpha3.set(alpha1, pixels);
        alpha3.set(alpha1, pixels * 2);
        return new ort.Tensor('float32', alpha3, [1, 3, height, width]);
    };
    Convert.squeezeAlpha = ({ dims, data: alpha3 }) => {
        const [height, width] = dims.slice(2);
        const pixels = width * height;
        const alpha1 = new Float32Array(pixels);
        let a1 = 0, a2 = pixels, a3 = pixels * 2;
        for (let i = 0; i < alpha3.length; i++)
            alpha1[i] += alpha3[a1++] / 3;
        for (let i = 0; i < alpha3.length; i++)
            alpha1[i] += alpha3[a2++] / 3;
        for (let i = 0; i < alpha3.length; i++)
            alpha1[i] += alpha3[a3++] / 3;
        return new ort.Tensor('float32', alpha1, [1, 1, height, width]);
    };
    const srgbToLinearLut = new Uint16Array(256);
    const linearToSrgbLut = new Float32Array(4096);
    for (let byte = 0, lastIdx = 0; byte < 256; byte++) {
        const srgb = byte / 255;
        const linear = srgb > 0.0392857 ? Math.pow((srgb + 0.055) / 1.055, 2.4) : srgb / 12.9232102;
        linearToSrgbLut.fill(srgb, lastIdx, lastIdx = (srgbToLinearLut[byte] = (linear * 4095) >> 0) + 1);
    }
    // for (let srgb = 0; srgb < 256; srgb++) {
    // 	const linear = srgbToLinearLut[srgb]
    // 	const value = (linearToSrgbLut[linear] * 255) >> 0
    // 	if (value !== srgb) throw new TypeError(`sRGB roundtrip fail: ${value} !== ${srgb} (= ${linear})`)
    // }
    Convert.toRgb = ({ width, height, data }, bgR = 1, bgG = 1, bgB = 1) => {
        const bgRgb = new Uint16Array([
            srgbToLinearLut[(bgR * 255) >> 0],
            srgbToLinearLut[(bgG * 255) >> 0],
            srgbToLinearLut[(bgB * 255) >> 0]
        ]);
        const pixels = width * height;
        const rgb = new Float32Array(pixels * 3);
        const alpha = new Float32Array(pixels);
        for (let i = 3, a = 0; i < data.length; i += 4)
            alpha[a++] = data[i] / 255;
        for (let channelIdx = 0; channelIdx < 3; channelIdx++) {
            const bgLinear = bgRgb[channelIdx];
            const channel = rgb.subarray(channelIdx * pixels);
            for (let idx = channelIdx, out = 0; idx < data.length; idx += 4, out++) {
                channel[out] = linearToSrgbLut[(bgLinear + (srgbToLinearLut[data[idx]] - bgLinear) * alpha[out]) >> 0];
            }
        }
        return new ort.Tensor('float32', rgb, [1, 3, height, width]);
    };
    Convert.rgbToImageData = ({ dims, data: rgb }, alpha = null) => {
        const [height, width] = dims.slice(2);
        const pixels = width * height;
        const data = new Uint8ClampedArray(pixels * 4);
        if (!alpha)
            data.fill(255); // alpha
        let r = 0, g = pixels, b = pixels * 2;
        for (let i = 0; i < data.length; i += 4)
            data[i] = rgb[r++] * 255;
        for (let i = 1; i < data.length; i += 4)
            data[i] = rgb[g++] * 255;
        for (let i = 2; i < data.length; i += 4)
            data[i] = rgb[b++] * 255;
        if (alpha) {
            const alphaData = alpha.data;
            for (let i = 3, a1 = 0, a2 = pixels, a3 = pixels * 2; i < data.length; i += 4)
                data[i] = (alphaData[a1++] + alphaData[a2++] + alphaData[a3++]) / 3 * 255;
        }
        return new ImageData(data, width, height);
    };
    Convert.bleedEdges = ({ dims, data: rgb }, { data: alpha }, threshold = 0.5) => {
        const width = dims[3];
        const height = dims[2];
        const pixels = width * height;
        // SoA (typed arrays) *much* faster than AoS here
        let numEdges = Math.ceil(pixels / 2);
        let numPixels = pixels;
        const buffer = new ArrayBuffer(numEdges * 32 + numPixels * 24);
        const edgesX = new Float64Array(buffer, 0, numEdges);
        const edgesY = new Float64Array(buffer, numEdges * 8, numEdges);
        const edgesR = new Float32Array(buffer, numEdges * 16, numEdges);
        const edgesG = new Float32Array(buffer, numEdges * 20, numEdges);
        const edgesB = new Float32Array(buffer, numEdges * 24, numEdges);
        const pixelsX = new Float64Array(buffer, numEdges * 32, numPixels);
        const pixelsY = new Float64Array(buffer, numEdges * 32 + numPixels * 8, numPixels);
        const pixelsI = new Float64Array(buffer, numEdges * 32 + numPixels * 16, numPixels);
        numEdges = 0;
        numPixels = 0;
        {
            const r = rgb.subarray(0);
            const g = rgb.subarray(pixels);
            const b = rgb.subarray(pixels * 2);
            for (let y = 0, i = 0; y < height; y++) {
                const hasAbove = y > 0;
                const hasBelow = y < height - 1;
                for (let x = 0; x < width; x++, i++) {
                    if (alpha[i] <= threshold) {
                        pixelsX[numPixels] = x;
                        pixelsY[numPixels] = y;
                        pixelsI[numPixels] = i;
                        numPixels++;
                    }
                    else if ((x > 0 && alpha[i - 1] <= threshold) || // left
                        (x < width - 1 && alpha[i + 1] <= threshold) || // right
                        (hasAbove && alpha[i - width] <= threshold) || // above
                        (hasBelow && alpha[i + width] <= threshold) // below
                    ) {
                        edgesX[numEdges] = x;
                        edgesY[numEdges] = y;
                        edgesR[numEdges] = r[i];
                        edgesG[numEdges] = g[i];
                        edgesB[numEdges] = b[i];
                        numEdges++;
                    }
                }
            }
        }
        const data = new Float32Array(rgb);
        if (numEdges > 0) {
            const r = data.subarray(0);
            const g = data.subarray(pixels);
            const b = data.subarray(pixels * 2);
            for (let p = 0; p < numPixels; p++) {
                const pixelX = pixelsX[p];
                const pixelY = pixelsY[p];
                let closestDistanceSquared = Infinity;
                let closestIndex = 0;
                for (let e = 0; e < numEdges; e++) {
                    const distanceSquared = (edgesX[e] - pixelX) ** 2 + (edgesY[e] - pixelY) ** 2;
                    if (distanceSquared < closestDistanceSquared) {
                        closestDistanceSquared = distanceSquared;
                        closestIndex = e;
                    }
                }
                const i = pixelsI[p];
                r[i] = edgesR[closestIndex];
                g[i] = edgesG[closestIndex];
                b[i] = edgesB[closestIndex];
            }
        }
        return new ort.Tensor('float32', data, [1, 3, height, width]);
    };
    Convert.batch = (...images) => {
        const dims = images[0].dims;
        const width = dims[3];
        const height = dims[2];
        const batch = new Float32Array(width * height * 3 * images.length);
        let offset = 0;
        for (const image of images) {
            if (image.dims[0] > 1)
                throw new TypeError('image already batched');
            batch.set(image.data, offset);
            offset += image.data.length;
        }
        return new ort.Tensor('float32', batch, [images.length, 3, height, width]);
    };
    Convert.unbatch = (batch) => {
        const dims = batch.dims;
        const width = dims[3];
        const height = dims[2];
        const pixels = width * height * 3;
        const bitmaps = [];
        let offset = 0;
        for (let i = 0; i < dims[0]; i++) {
            bitmaps[i] = new ort.Tensor('float32', batch.data.slice(offset, offset += pixels), [1, 3, height, width]);
        }
        return bitmaps;
    };
})(Convert || (Convert = {}));
/// <reference path="convert.ts" />
var Jobs;
(function (Jobs) {
    var GlBlitter = Convert.GlBlitter;
    Jobs.spawn = (init) => new Promise((accept, reject) => {
        if (init.getState() === 'stop') {
            init.reportCallback({ type: 'aborted' });
            accept(null);
            return;
        }
        const store = {
            ...init,
            stop: false,
            timer: new Timer(),
            yield: () => new Promise(async (accept) => {
                if (store.stop)
                    return;
                let state = store.getState();
                if (state === 'paused') {
                    store.reportCallback({ type: 'paused' });
                    while (state === 'paused')
                        state = await store.getNextState();
                    if (state === 'stop' || store.stop)
                        return;
                    store.reportCallback({ type: 'unpaused' });
                }
                if (state !== 'stop' && !store.stop)
                    accept();
            }),
            wrap: (promise) => new Promise(async (accept, reject) => {
                await store.yield();
                await promise.then(async (value) => {
                    await store.yield();
                    accept(value);
                }, async (value) => {
                    await store.yield();
                    reject(value);
                });
            }),
            output: undefined
        };
        const onFinished = (report) => {
            console.info(`${'―'.repeat(4)} unlimited:waifu2x job ${report.type === 'aborted' ? 'aborted' : report.type === 'errored' ? 'FAILED' : 'completed'} ${'―'.repeat(91 + (report.type === 'aborted' ? 2 : report.type === 'errored' ? 3 : 0))}`);
            console.info(`· Input: ${store.initParams.image.width}x${store.initParams.image.height} (${store.initParams.image.width * store.initParams.image.height}px)`);
            console.info(`· Output: ${store.initParams.image.width * store.initParams.scale}x${store.initParams.image.height * store.initParams.scale} (${store.initParams.image.width * store.initParams.scale * store.initParams.image.height * store.initParams.scale}px)`);
            console.info(`· Model: ${store.initParams.model.name}.${store.initParams.model.style}`);
            console.info(`· Denoise: ${store.initParams.model.noise}`);
            console.info(`· Scale: ${store.initParams.scale}`);
            console.info(`· Tile size: ${store.initParams.tileSize}`);
            console.info(`· TTA level: ${store.initParams.ttaLevel}`);
            console.info(`· Alpha: ${store.initParams.alphaChannel}`);
            console.info(`· Threads: ${!!SharedArrayBuffer ? ort.env.wasm.numThreads : '1 (no SharedArrayBuffer)'}`);
            console.info('\n');
            store.timer.printSummary();
            console.info('―'.repeat(128));
            return report;
        };
        // avoid ABA
        store.getNextState().then(async function watchForStop(newState) {
            if (newState === 'stop') {
                if (store.stop)
                    return;
                store.stop = true;
                store.reportCallback(onFinished({ type: 'aborted' }));
                accept(null);
            }
            else {
                store.getNextState().then(watchForStop);
            }
        });
        run(store).then(output => {
            store.stop = true;
            store.reportCallback(onFinished({ type: 'completed', value: output }));
            accept(output);
        }, error => {
            store.stop = true;
            store.reportCallback(onFinished({ type: 'errored', error }));
            reject(error);
        });
    });
    function collectTiles(width, height, tileSize) {
        const tiles = [];
        for (let tileY = 0; tileY < height; tileY += tileSize) {
            for (let tileX = 0; tileX < width; tileX += tileSize) {
                tiles.push([tileX, tileY]);
            }
        }
        return tiles;
    }
    function calculateTileMetrics(width, height, tileX, tileY, tileSize, context) {
        const tileWidth = Math.min(tileSize, width - tileX);
        const tileHeight = Math.min(tileSize, height - tileY);
        const tileWidthAdjusted = Math.ceil(tileWidth / 4) * 4;
        const tileHeightAdjusted = Math.ceil(tileHeight / 4) * 4;
        const tilePixels = (tileWidthAdjusted + context * 2) * (tileHeightAdjusted + context * 2);
        const tilePadding = {
            left: Math.max(context - tileX, 0),
            right: Math.max(tileX + tileWidthAdjusted + context - width, 0),
            top: Math.max(context - tileY, 0),
            bottom: Math.max(tileY + tileHeightAdjusted + context - height, 0)
        };
        const captureX = tileX - context + tilePadding.left;
        const captureY = tileY - context + tilePadding.top;
        const captureW = Math.min(tileX + tileWidthAdjusted + context, width) - captureX;
        const captureH = Math.min(tileY + tileHeightAdjusted + context, height) - captureY;
        return { tileWidth, tileHeight, tileWidthAdjusted, tileHeightAdjusted, tilePixels, tilePadding, captureX, captureY, captureW, captureH };
    }
    function calculateTotalPixels(tiles, width, height, tileSize, context) {
        let pixelsTotal = 0;
        for (const [tileX, tileY] of tiles)
            pixelsTotal += calculateTileMetrics(width, height, tileX, tileY, tileSize, context).tilePixels;
        return pixelsTotal;
    }
    const run = async (store) => {
        const { utils, initParams: { image: input, image: { width: inputWidth, height: inputHeight }, output, scale, tileSize, alphaChannel, backgroundColor: [bgR, bgG, bgB], alphaThreshold, model, model: { context }, antialias, ttaLevel }, liveParams, timer, wrap } = store;
        const outputWidth = inputWidth * scale;
        const outputHeight = inputHeight * scale;
        timer.push('run');
        timer.push('canvas');
        timer.push('getContext');
        const ctx = output.getContext('2d');
        timer.transition('clear');
        ctx.clearRect(0, 0, outputWidth, outputHeight);
        timer.transition('backdrop');
        ctx.filter = `brightness(75%) blur(${Math.max(outputWidth, outputHeight) * 0.01}px)`;
        ctx.drawImage(input, 0, 0, outputWidth, outputHeight);
        ctx.filter = 'none';
        timer.pop();
        timer.transition('glBlitter');
        store.output = new GlBlitter(outputWidth, outputHeight);
        timer.transition('tiledRender');
        timer.push('collectTiles');
        const tiles = collectTiles(inputWidth, inputHeight, tileSize);
        const tilesTotal = tiles.length;
        let tilesCompleted = 0;
        timer.push('calculateTotalPixels');
        const pixelsTotal = calculateTotalPixels(tiles, inputWidth, inputHeight, tileSize, context);
        let pixelsCompleted = 0;
        timer.transition('newScratch');
        const scratch = new Uint8ClampedArray((Math.ceil(tileSize / 4) * 4 + context * 2) ** 2 * 4);
        await store.yield();
        timer.pop();
        // TODO move these stages into separate functions, maybe support batching
        while (tiles.length > 0) {
            timer.push('tile');
            timer.push('pick');
            let tileX = 0, tileY = 0;
            const focus = liveParams.tileFocus();
            if (focus) {
                timer.push('focus');
                const [focusX, focusY] = focus;
                let closestDistanceSquared = Infinity;
                let closestTileIndex = 0;
                for (let i = 0; i < tiles.length; i++) {
                    const [tileX, tileY] = tiles[i];
                    const distanceSquared = (tileX + tileSize / 2 - focusX / scale) ** 2 + (tileY + tileSize / 2 - focusY / scale) ** 2;
                    if (distanceSquared < closestDistanceSquared) {
                        closestDistanceSquared = distanceSquared;
                        closestTileIndex = i;
                    }
                }
                [tileX, tileY] = tiles.splice(closestTileIndex, 1)[0];
                timer.pop();
            }
            else {
                [tileX, tileY] = liveParams.tileRandom()
                    ? tiles.splice(Math.floor(Math.random() * tiles.length), 1)[0]
                    : tiles.shift();
            }
            timer.transition('calculateTileMetrics');
            const metrics = calculateTileMetrics(inputWidth, inputHeight, tileX, tileY, tileSize, context);
            timer.transition('reportCallbackStarted');
            store.reportCallback({
                type: 'tile-started',
                tilesTotal: tilesTotal,
                tilesCompleted: tilesCompleted,
                pixelsTotal: pixelsTotal,
                pixelsCompleted: pixelsCompleted,
                pixelsStarted: metrics.tilePixels
            });
            timer.transition('indicator');
            const tileMinDim = Math.min(metrics.tileWidth, metrics.tileHeight);
            ctx.strokeStyle = 'deepskyblue';
            ctx.lineWidth = Math.floor(Math.min(Math.max(tileMinDim / 10 * scale, 2), tileMinDim / 3 * scale));
            ctx.strokeRect(tileX * scale + ctx.lineWidth / 2, tileY * scale + ctx.lineWidth / 2, metrics.tileWidth * scale - ctx.lineWidth, metrics.tileHeight * scale - ctx.lineWidth);
            timer.transition('readPixels');
            const capture = new ImageData(scratch.subarray(0, metrics.captureW * metrics.captureH * 4), metrics.captureW, metrics.captureH);
            Convert.readPixels(await wrap(createImageBitmap(input, metrics.captureX, metrics.captureY, metrics.captureW, metrics.captureH, { premultiplyAlpha: 'none' })), capture.data);
            timer.transition('process');
            let uncropped;
            if (alphaChannel) {
                timer.push('toRgbAlpha');
                let [rgb, alpha1] = Convert.toRgbAlpha(capture);
                timer.transition('bleedEdges');
                rgb = Convert.bleedEdges(rgb, alpha1, alphaThreshold);
                timer.transition('stretchAlpha');
                let alpha3 = Convert.stretchAlpha(alpha1);
                timer.transition('pad');
                timer.push('rgb');
                rgb = await wrap(utils.pad(rgb, metrics.tilePadding));
                timer.transition('alpha');
                alpha3 = await wrap(utils.pad(alpha3, metrics.tilePadding));
                timer.pop();
                if (ttaLevel > 0) {
                    timer.transition('ttaSplit');
                    timer.push('rgb');
                    rgb = await wrap(utils.tta_split(rgb, ttaLevel));
                    timer.transition('alpha');
                    alpha3 = await wrap(utils.tta_split(alpha3, ttaLevel));
                    timer.pop();
                }
                if (antialias) {
                    timer.transition('antialias');
                    timer.push('rgb');
                    rgb = await wrap(utils.antialias(rgb));
                    timer.transition('alpha');
                    alpha3 = await wrap(utils.antialias(alpha3));
                    timer.pop();
                }
                timer.transition('model');
                // TTA does its own batching
                if (ttaLevel === 0) {
                    timer.push('batch');
                    let batch = Convert.batch(rgb, alpha3);
                    timer.transition('run');
                    batch = await wrap(model.run(batch));
                    timer.transition('unbatch');
                    [rgb, alpha3] = Convert.unbatch(batch);
                    timer.pop();
                }
                else {
                    timer.push('rgb');
                    rgb = await wrap(model.run(rgb));
                    timer.transition('alpha');
                    alpha3 = await wrap(model.run(alpha3));
                    timer.pop();
                }
                if (ttaLevel > 0) {
                    timer.transition('ttaMerge');
                    timer.push('rgb');
                    rgb = await wrap(utils.tta_merge(rgb, ttaLevel));
                    timer.transition('alpha');
                    alpha3 = await wrap(utils.tta_merge(alpha3, ttaLevel));
                    timer.pop();
                }
                timer.transition('rgbToImageData');
                uncropped = Convert.rgbToImageData(rgb, alpha3);
                timer.pop();
            }
            else {
                timer.push('toRgb');
                let rgb = Convert.toRgb(capture, bgR, bgG, bgB);
                timer.transition('pad');
                rgb = await wrap(utils.pad(rgb, metrics.tilePadding));
                if (ttaLevel > 0) {
                    timer.transition('ttaSplit');
                    rgb = await wrap(utils.tta_split(rgb, ttaLevel));
                }
                if (antialias) {
                    timer.transition('antialias');
                    rgb = await wrap(utils.antialias(rgb));
                }
                timer.transition('model');
                rgb = await wrap(model.run(rgb));
                if (ttaLevel > 0) {
                    timer.transition('ttaMerge');
                    rgb = await wrap(utils.tta_merge(rgb, ttaLevel));
                }
                timer.transition('rgbToImageData');
                uncropped = Convert.rgbToImageData(rgb);
                timer.pop();
            }
            timer.transition('crop');
            // crop out all the padding and model-specific artifacts
            const regionDiffX = uncropped.width - metrics.tileWidthAdjusted * scale;
            const regionDiffY = uncropped.height - metrics.tileHeightAdjusted * scale;
            const cropped = await wrap(createImageBitmap(uncropped, regionDiffX / 2, regionDiffY / 2, metrics.tileWidth * scale, metrics.tileHeight * scale, { premultiplyAlpha: 'none' }));
            timer.transition('writePixels');
            store.output.writePixels(cropped, tileX * scale, tileY * scale);
            timer.transition('clearRect');
            ctx.clearRect(tileX * scale, tileY * scale, metrics.tileWidth * scale, metrics.tileHeight * scale);
            timer.transition('drawImage');
            ctx.drawImage(cropped, tileX * scale, tileY * scale);
            timer.transition('updateCounters');
            tilesCompleted++;
            pixelsCompleted += metrics.tilePixels;
            timer.transition('reportCallbackCompleted');
            store.reportCallback({ type: 'tile-completed', tilesTotal, tilesCompleted, pixelsTotal, pixelsCompleted, pixelsJustCompleted: metrics.tilePixels });
            timer.pop();
            timer.pop();
        }
        timer.transition('toImageData');
        const outputData = store.output.toImageData();
        timer.transition('outputBitmap');
        const outputBitmap = await wrap(createImageBitmap(outputData, { premultiplyAlpha: 'none' }));
        timer.pop();
        timer.pop();
        return outputBitmap;
    };
})(Jobs || (Jobs = {}));
/// <reference path="models.ts" />
/// <reference path="interface/interface.ts" />
/// <reference path="interface/message.ts" />
/// <reference path="interface/jobStatus.ts" />
/// <reference path="jobs/jobs.ts" />
/// <reference path="metadata.ts" />
(async () => {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = true;
    // noinspection ES6MissingAwait
    UserInterface.setMessageWorking('Loading utility models');
    const utils = await Utils.load();
    await UserInterface.setStartCallback(async (initParams, liveParams) => {
        const modelSuffix = `${initParams.model.name}_${initParams.model.style}_${ModelMetadata.getModelBasename(initParams.scale, initParams.model.noise)}`;
        const mapState = (state) => {
            if (state === 'preinit' || state === 'busy') {
                return 'stop';
            }
            else {
                return state;
            }
        };
        let status = null;
        const outputBitmap = await Jobs.spawn({
            utils,
            initParams,
            liveParams,
            getState: () => mapState(UserInterface.getState()),
            getNextState: () => UserInterface.getNextState().then(mapState),
            reportCallback: async (report) => {
                console.log(report);
                switch (report.type) {
                    case 'start':
                        break;
                    case 'paused':
                        if (status)
                            status.pause();
                        break;
                    case 'unpaused':
                        if (status)
                            status.unpause();
                        break;
                    case 'tile-started':
                        if (!status) {
                            const textNode = document.createTextNode('UwU');
                            await UserInterface.setMessageWorking(textNode);
                            status = new JobStatusIndicator(report.tilesTotal, report.pixelsTotal, text => textNode.textContent = text);
                        }
                        status.reportTileStarted(report.pixelsStarted);
                        break;
                    case 'tile-completed':
                        status.reportTileCompleted(report.pixelsJustCompleted);
                        break;
                    case 'completed':
                        break;
                    case 'aborted':
                        break;
                    case 'errored':
                        break;
                }
            }
        });
        if (outputBitmap) {
            const downloadLink = document.createElement('a');
            downloadLink.setAttribute('href', 'javascript:void(0)');
            downloadLink.appendChild(document.createTextNode('download'));
            downloadLink.addEventListener('click', async (e) => {
                e.preventDefault();
                const outputBitmap = UserInterface.getOutputBitmap();
                if (!outputBitmap || UserInterface.getState() !== 'stop')
                    return;
                await UserInterface.setState('busy');
                const canvas = new OffscreenCanvas(outputBitmap.width, outputBitmap.height);
                canvas.getContext('bitmaprenderer').transferFromImageBitmap(outputBitmap);
                const blob = await canvas.convertToBlob({ type: 'image/png' });
                await UserInterface.setOutputBitmap(canvas.transferToImageBitmap());
                try {
                    const handle = await showSaveFilePicker({ types: [{ accept: { 'image/png': ['.png'] } }], suggestedName: UserInterface.getOutputFilename() + `_waifu2x_${modelSuffix}` });
                    const stream = await handle.createWritable({});
                    await stream.write(blob);
                    await stream.close();
                }
                catch (e) {
                    if (e instanceof DOMException && e.name === 'AbortError') { }
                    else
                        throw e;
                }
                finally {
                    await UserInterface.setState('stop');
                }
            });
            await UserInterface.setMessageFace('give', status.conclude(), '—', downloadLink);
            status = null;
            await UserInterface.setOutputBitmap(outputBitmap);
            await UserInterface.setState('stop');
        }
    });
    await UserInterface.setMessageFace('neutral');
})();
class Throttler {
    threshold;
    lastYield;
    constructor(threshold = 50) {
        this.threshold = threshold;
        this.lastYield = performance.now();
    }
    'yield'() {
        return new Promise(requestAnimationFrame);
    }
    async tick() {
        if (performance.now() - this.lastYield > this.threshold) {
            this.lastYield = await this.yield();
        }
    }
}
class Timer {
    history = new Map();
    keyStack = [];
    startStack = [];
    extraStack = [];
    constructor() { }
    push(name, extra) {
        this.keyStack.push(name);
        const key = this.keyStack.join('.');
        this.history.set(key, this.history.get(key) ?? []);
        this.startStack.push(performance.now());
        this.extraStack.push(extra);
    }
    pop(setExtra) {
        const now = performance.now();
        const key = this.keyStack.join('.');
        this.keyStack.pop();
        const times = this.history.get(key);
        const duration = now - this.startStack.pop();
        const poppedExtra = this.extraStack.pop();
        times.push({ duration, extra: setExtra !== undefined ? setExtra : poppedExtra });
        return duration;
    }
    transition(name, extra) {
        const duration = this.pop();
        this.push(name, extra);
        return duration;
    }
    printSummary() {
        while (this.keyStack.length > 0)
            this.pop();
        for (const [key, entries] of this.history) {
            const level = '\t'.repeat(key.match(/\./g)?.length ?? 0);
            if (entries.length > 1) {
                let total = 0, min = Infinity, max = 0;
                for (const entry of entries) {
                    total += entry.duration;
                    if (entry.duration < min)
                        min = entry.duration;
                    if (entry.duration > max)
                        max = entry.duration;
                }
                console.groupCollapsed(`${level}→ %s: ${entries.length} occurrences; min ${min.toFixed(2)}ms, max ${max.toFixed(2)}ms, avg ${(total / entries.length).toFixed(2)}ms, total ${total.toFixed(2)}ms`, key);
            }
            else {
                console.groupCollapsed(`${level}→ %s: ${entries[0].duration.toFixed(2)}ms`, key);
            }
            for (const entry of entries) {
                if (entry.extra) {
                    console.info(`${level}↑ ${entry.duration.toFixed(2)}ms → %o`, entry.extra);
                }
                else if (entries.length > 1) {
                    console.info(`${level}↑ ${entry.duration.toFixed(2)}ms`);
                }
            }
            console.groupEnd();
        }
    }
}
//# sourceMappingURL=script.js.map