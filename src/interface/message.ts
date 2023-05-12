/// <reference path="../documentLoaded.ts" />
/// <reference path="../createSignal.ts" />

namespace UserInterface {
	const _message = documentLoaded.then(() => document.getElementById('message') as HTMLDivElement)

	let [onMessageChanged, messageChanged] = createSignal<void>()

	export const setMessage = async (...contents: (Node | string)[]) => {
		const message = await _message
		message.replaceChildren(...contents)
		onMessageChanged()
	}

	const faces = {
		neutral: '( ・∀・)',
		happy: '(ﾟ∀ﾟ)',
		good: '( ・∀・)b',
		give: '( ・∀・)つ',
		shock: '(・A・)',
		panic: '(・A・)!!',
		unamused: '(-_-)'
	}

	export const setMessageFace = (face: keyof typeof faces, ...contents: (Node | string)[]) =>
		setMessage(`${faces[face]}${contents.length > 0 ? '　' : ''}`, ...contents)

	export const setMessageWorking = async (...contents: (Node | string)[]) => {
		await _message

		let currentFace = '( ・∀・)φ　 ', nextFace = '( ・∀・) φ　'
		const faceAnimationNode = document.createTextNode(currentFace)
		const faceInterval = setInterval(() => {
			const toSet = nextFace
			faceAnimationNode.textContent = toSet
			nextFace = currentFace
			currentFace = toSet
		}, 500)

		let dotPos = 0
		const numDots = 3
		const dotsAnimationNode = document.createTextNode(` ${'.'.repeat(numDots)}`)
		const dotsInterval = setInterval(() => {
			dotPos += 1
			dotPos %= numDots + 1
			dotsAnimationNode.textContent = `${'.'.repeat(dotPos)} ${'.'.repeat(numDots - dotPos)}`
		}, 250)

		await setMessage(faceAnimationNode, ...contents, dotsAnimationNode)
		messageChanged().then(() => clearInterval(faceInterval))
		messageChanged().then(() => clearInterval(dotsInterval))
	}

	window.addEventListener('unhandledrejection', async e => {
		await setMessageFace('unamused', 'Unexpected error: ', '' + e.reason)
	})
}
