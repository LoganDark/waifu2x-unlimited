class Throttler {
	private lastYield: number

	public constructor(public threshold: number = 50) {
		this.lastYield = performance.now()
	}

	public 'yield'() {
		return new Promise<DOMHighResTimeStamp>(requestAnimationFrame)
	}

	public async tick() {
		if (performance.now() - this.lastYield > this.threshold) {
			this.lastYield = await this.yield()
		}
	}
}
