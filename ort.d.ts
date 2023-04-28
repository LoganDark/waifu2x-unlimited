declare global {
	const ort: typeof import('onnxruntime-common')

	namespace ort {
		type InferenceSession = import('onnxruntime-common').InferenceSession
		type TypedTensor<T> = import('onnxruntime-common').TypedTensor<T>
	}
}

export {}
