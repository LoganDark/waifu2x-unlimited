class JobStatusIndicator {
	private _start: DOMHighResTimeStamp
	private _raf: number | null
	private _completedTiles = 0
	private _completedPixels = 0
	private _tileLastCompleted: DOMHighResTimeStamp | null = null
	private _largestTilePixels = 0
	private _currentTilePixels = 0
	private _pausedAt: DOMHighResTimeStamp | null = null

	public constructor(
		private _totalTiles: number,
		private _totalPixels: number,
		private _callback: (status: string) => void
	) {
		this._start = performance.now()
		this._raf = requestAnimationFrame(this.tick)
	}

	public tick = (now: DOMHighResTimeStamp) => {
		try {
			const since = now - this._start

			// precision so that every tile completed results in a percentage change
			const precision = Math.max(Math.log10(this._totalTiles / 100), 0)
			let message = `${this._completedTiles}/${this._totalTiles} (${(this._completedPixels / this._totalPixels * 100).toFixed(precision)}% complete)`

			if (this._tileLastCompleted !== null) {
				const forCurrentPixels = this._tileLastCompleted - this._start
				const currentLargestTiles = this._completedPixels / this._largestTilePixels
				const perLargestTile = forCurrentPixels / currentLargestTiles

				message += ` (${perLargestTile.toFixed(1)}ms/t`

				if (perLargestTile > 1500) {
					if (this._completedTiles < this._totalTiles) {
						const perCurrentTile = this._currentTilePixels / this._largestTilePixels * perLargestTile
						const estimatedProgress = Math.min((since - forCurrentPixels) / perCurrentTile * 0.95, 1)
						// precision so that every 100ms results in a percentage change
						const precision = Math.max(Math.log10(perLargestTile / 1000), 0)
						message += `; ${(estimatedProgress * 100).toFixed(precision)}%${estimatedProgress === 1 ? '?' : ''})`
					} else {
						message += ')'
					}
				} else {
					message += `; ${(1000 / perLargestTile).toFixed(2)}t/s)`
				}
			}

			this._callback(message)
		} finally {
			this._raf = requestAnimationFrame(this.tick)
		}
	}

	public pause() {
		if (this._raf === null) throw new TypeError('already paused')
		cancelAnimationFrame(this._raf)
		this._pausedAt = performance.now()
	}

	public unpause() {
		if (this._pausedAt === null) throw new TypeError('not paused')
		const now = performance.now()
		const duration = now - this._pausedAt
		this._start += duration
		if (this._tileLastCompleted !== null) this._tileLastCompleted += duration
		this._pausedAt = null
		this._raf = requestAnimationFrame(this.tick)
	}

	public reportTileStarted(pixels: number) {
		this._currentTilePixels = pixels
	}

	public reportTileCompleted(pixels: number) {
		this._tileLastCompleted = performance.now()
		this._completedTiles++
		this._completedPixels += pixels

		if (pixels > this._largestTilePixels)
			this._largestTilePixels = pixels
	}

	public conclude() {
		this.pause()
		const duration = this._pausedAt! - this._start
		return `${duration.toFixed(2)}ms`
	}
}
