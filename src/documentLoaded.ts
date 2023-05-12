const documentLoaded = document.readyState != 'loading'
	? Promise.resolve()
	: new Promise<void>(resolve => document.addEventListener('DOMContentLoaded', () => resolve()))
