const createSignal = <T>() => {
	let fire: (value: T) => void, next = (function refresh() {
		return new Promise<T>(resolve =>
			fire = (value) => {
				next = refresh()
				resolve(value)
			})
	})()

	return [(value: T) => fire(value), () => next] as const
}
