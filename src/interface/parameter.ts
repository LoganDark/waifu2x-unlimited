/// <reference path="../metadata.ts" />

namespace UserInterface {
	export abstract class Parameter<T, E extends HTMLElement> {
		protected _defaultValue: T

		protected constructor(public element: E, protected _onValueChanged: (newValue: T) => void = () => {}) {
			[this._fireValueChanged, this.getNextValue] = createSignal()
			this._defaultValue = this._getValue()
		}

		protected _init() {
			let savedValue: T | null = null
			try { savedValue = this._getSavedValue() } catch {}
			if (savedValue !== null && savedValue !== this._getValue()) this._setValue(savedValue)

			const value = savedValue === null ? this._getValue() : savedValue
			this._setSavedValue(value)
			this._onValueChanged(value)
		}

		protected abstract _getSavedValue(): T | null

		protected abstract _setSavedValue(newValue: T): void

		protected abstract _getValue(): T

		protected abstract _setValue(newValue: T): boolean

		private _fireValueChanged: (newValue: T) => void

		public getNextValue: () => Promise<T>

		public get value() { return this._getValue() }

		public set value(newValue: T) {
			if (this._setValue(newValue)) {
				this._setSavedValue(newValue)
				//this._onValueChanged(newValue)
				this._fireValueChanged(newValue)
			}
		}

		public update() {
			const newValue = this.value
			this._setSavedValue(newValue)
			this._onValueChanged(newValue)
			this._fireValueChanged(newValue)
		}
	}

	export namespace Parameter {
		export interface Disableable {
			get disabled(): boolean

			set disabled(disabled: boolean)
		}

		export class FilePicker extends Parameter<File | null, HTMLInputElement> implements Disableable {
			public constructor(picker: HTMLInputElement, _onValueChanged?: (newValue: File | null) => void) {
				super(picker, _onValueChanged)
				picker.addEventListener('change', () => this.update())
				this._init()
			}

			protected _getSavedValue(): File | null { return null }

			protected _setSavedValue(newValue: File | null): void {}

			protected _getValue(): File | null { return this.element.files?.item(0) ?? null }

			protected _setValue(newValue: File | null): boolean { return newValue == this._getValue() }

			public get disabled(): boolean { return this.element.disabled }

			public set disabled(disabled: boolean) { this.element.disabled = disabled }
		}

		export abstract class Select<T> extends Parameter<T, HTMLSelectElement> implements Disableable {
			public constructor(select: HTMLSelectElement, private key: string, _onValueChanged?: (newValue: T) => void) {
				super(select, _onValueChanged)
				select.addEventListener('change', () => this.update())
				this._init()
			}

			protected abstract _fromString(value: string): T | null

			protected abstract _toString(value: T): string

			protected _getSavedValue(): T | null {
				let stored: string | null = null

				try {
					stored = localStorage.getItem(this.key)
				} catch {}

				return stored !== null ? this._fromString(stored) : null
			}

			protected _setSavedValue(newValue: T) {
				const toStore = this._toString(newValue)

				try {
					localStorage.setItem(this.key, toStore)
				} catch {}
			}

			protected _getValue(): T {
				const value = this._fromString(this.element.value)

				if (value === null) {
					throw new TypeError('current element value is invalid')
				} else {
					return value
				}
			}

			protected _setValue(newValue: T): boolean {
				const value = this._toString(newValue)

				if (value !== this.element.value) {
					this.element.value = value
					return true
				} else {
					return false
				}
			}

			public get disabled(): boolean { return this.element.disabled }

			public set disabled(disabled: boolean) { this.element.disabled = disabled }
		}

		export class Checkbox extends Parameter<boolean, HTMLInputElement> implements Disableable {
			public constructor(checkbox: HTMLInputElement, private key: string, _onValueChanged?: (newValue: boolean) => void) {
				if (checkbox.type !== 'checkbox') throw new TypeError('expected checkbox')
				super(checkbox, _onValueChanged)
				checkbox.addEventListener('change', () => this.update())
				this._init()
			}

			protected _getSavedValue(): boolean | null {
				try {
					const stored = localStorage.getItem(this.key)
					return stored === 'true' ? true : stored === 'false' ? false : null
				} catch {}

				return null
			}

			protected _setSavedValue(newValue: boolean) {
				try {
					localStorage.setItem(this.key, newValue ? 'true' : 'false')
				} catch {}
			}

			protected _getValue(): boolean {
				return this.element.checked
			}

			protected _setValue(newValue: boolean): boolean {
				if (newValue !== this.element.checked) {
					this.element.checked = newValue
					return true
				} else {
					return false
				}
			}

			public get disabled(): boolean { return this.element.disabled }

			public set disabled(disabled: boolean) { this.element.disabled = disabled }
		}

		export class IntegerInput extends Parameter<number, HTMLInputElement> implements Disableable {
			public constructor(input: HTMLInputElement, private key: string, _onValueChanged?: (newValue: number) => void) {
				if (input.type !== 'number') throw new TypeError('expected number input')
				super(input, _onValueChanged)
				input.addEventListener('change', () => this.update())
				this._init()
			}

			protected _getSavedValue(): number | null {
				const stored = localStorage.getItem(this.key)
				const value = Number(stored ?? '!')
				if (!Number.isInteger(value)) return null
				return value
			}

			protected _setSavedValue(newValue: number) {
				try {
					localStorage.setItem(this.key, newValue.toString())
				} catch {}
			}

			protected _getValue(): number {
				const value = Number(this.element.value)

				if (!Number.isInteger(value)) {
					this._setValue(this._defaultValue)
					return this._defaultValue
				}

				return value
			}

			protected _setValue(newValue: number): boolean {
				const toSet = newValue.toString()

				if (toSet !== this.element.value) {
					this.element.value = toSet
					return true
				} else {
					return false
				}
			}

			public get disabled(): boolean { return this.element.disabled }

			public set disabled(disabled: boolean) { this.element.disabled = disabled }
		}

		export class NumberInput extends Parameter<number, HTMLInputElement> implements Disableable {
			public constructor(input: HTMLInputElement, private key: string, _onValueChanged?: (newValue: number) => void) {
				if (input.type !== 'number') throw new TypeError('expected number input')
				super(input, _onValueChanged)
				input.addEventListener('change', () => this.update())
				this._init()
			}

			protected _getSavedValue(): number | null {
				const stored = localStorage.getItem(this.key)
				const value = Number(stored ?? '!')
				if (!Number.isFinite(value)) return null
				return value
			}

			protected _setSavedValue(newValue: number) {
				try {
					localStorage.setItem(this.key, newValue.toString())
				} catch {}
			}

			protected _getValue(): number {
				const value = Number(this.element.value)

				if (!Number.isFinite(value)) {
					this._setValue(this._defaultValue)
					return this._defaultValue
				}

				return value
			}

			protected _setValue(newValue: number): boolean {
				const toSet = newValue.toString()

				if (toSet !== this.element.value) {
					this.element.value = toSet
					return true
				} else {
					return false
				}
			}

			public get disabled(): boolean { return this.element.disabled }

			public set disabled(disabled: boolean) { this.element.disabled = disabled }
		}

		export class ModelSelect extends Select<string> {
			protected _fromString(value: string): string | null {
				const [model, style] = value.split('.')
				if (ModelMetadata.models[model]?.styles?.has(style)) return value
				return null
			}

			protected _toString(value: string): string { return value }
		}

		export class IntegerSelect extends Select<number> {
			protected _fromString(value: string): number | null {
				const parsed = parseInt(value)
				return parsed !== parsed ? null : parsed
			}

			protected _toString(value: number): string {
				return value.toString()
			}
		}

		export class NoiseLevelSelect extends Select<Utils.NoiseLevel> {
			protected _fromString(value: string): Utils.NoiseLevel | null {
				return ['-1', '0', '1', '2', '3'].indexOf(value) > -1 ? +value as Utils.NoiseLevel : null
			}

			protected _toString(value: Utils.NoiseLevel): string {
				return value.toString()
			}
		}

		export class TtaLevelSelect extends Select<Utils.TtaLevel> {
			protected _fromString(value: string): Utils.TtaLevel | null {
				return ['0', '2', '4'].indexOf(value) > -1 ? +value as Utils.TtaLevel : null
			}

			protected _toString(value: Utils.TtaLevel): string {
				return value.toString()
			}
		}

		export class ColorInput extends Parameter<number, HTMLInputElement> implements Disableable {
			public constructor(input: HTMLInputElement, private key: string, _onValueChanged?: (newValue: number) => void) {
				if (input.type !== 'color') throw new TypeError('expected color input')
				super(input, _onValueChanged)
				input.addEventListener('change', () => this.update())
				this._init()
			}

			protected _getSavedValue(): number | null {
				const stored = localStorage.getItem(this.key)
				const value = parseInt(stored?.match(/^#([0-9a-f]{6})$/i)?.[1] ?? '!', 16)
				if (!Number.isFinite(value)) return null
				return value
			}

			protected _setSavedValue(newValue: number) {
				try {
					localStorage.setItem(this.key, `#${newValue.toString(16).padStart(6, '0')}`)
				} catch {}
			}

			protected _getValue(): number {
				const value = parseInt(this.element.value.match(/^#([0-9a-f]{6})$/i)?.[1] ?? '!', 16)

				if (!Number.isFinite(value)) {
					this._setValue(this._defaultValue)
					return this._defaultValue
				}

				return value
			}

			protected _setValue(newValue: number): boolean {
				const toSet = `#${newValue.toString(16).padStart(6, '0')}`

				if (toSet.toLowerCase() !== this.element.value.toLowerCase()) {
					this.element.value = toSet
					return true
				} else {
					return false
				}
			}

			public get disabled(): boolean { return this.element.disabled }

			public set disabled(disabled: boolean) { this.element.disabled = disabled }
		}
	}
}
