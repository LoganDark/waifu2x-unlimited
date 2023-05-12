class Timer {
	private history = new Map<string, Timer.Entry<any>[]>()
	private keyStack: string[] = []
	private startStack: number[] = []
	private extraStack: any[] = []

	public constructor() {}

	public push(name: string, extra?: any) {
		this.keyStack.push(name)
		const key = this.keyStack.join('.')
		this.history.set(key, this.history.get(key) ?? [])
		this.startStack.push(performance.now())
		this.extraStack.push(extra)
	}

	public pop(setExtra?: any) {
		const now = performance.now()
		const key = this.keyStack.join('.')
		this.keyStack.pop()
		const times = this.history.get(key)!
		const duration = now - this.startStack.pop()!
		const poppedExtra = this.extraStack.pop()!
		times.push({duration, extra: setExtra !== undefined ? setExtra : poppedExtra})
		return duration
	}

	public transition(name: string, extra?: any) {
		const duration = this.pop()
		this.push(name, extra)
		return duration
	}

	public printSummary() {
		while (this.keyStack.length > 0) this.pop()

		for (const [key, entries] of this.history) {
			const level = '\t'.repeat(key.match(/\./g)?.length ?? 0)

			if (entries.length > 1) {
				let total = 0, min = Infinity, max = 0
				for (const entry of entries) {
					total += entry.duration
					if (entry.duration < min) min = entry.duration
					if (entry.duration > max) max = entry.duration
				}

				console.groupCollapsed(`${level}→ %s: ${entries.length} occurrences; min ${min.toFixed(2)}ms, max ${max.toFixed(2)}ms, avg ${(total / entries.length).toFixed(2)}ms, total ${total.toFixed(2)}ms`, key)
			} else {
				console.groupCollapsed(`${level}→ %s: ${entries[0].duration.toFixed(2)}ms`, key)
			}

			for (const entry of entries) {
				if (entry.extra) {
					console.info(`${level}↑ ${entry.duration.toFixed(2)}ms → %o`, entry.extra)
				} else if (entries.length > 1) {
					console.info(`${level}↑ ${entry.duration.toFixed(2)}ms`)
				}
			}

			console.groupEnd()
		}
	}
}

namespace Timer {
	export interface Entry<E> {
		duration: number,
		extra: E
	}
}
