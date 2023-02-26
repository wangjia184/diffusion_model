/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// Workaround for: https://github.com/bazelbuild/rules_nodejs/issues/1265
/// <reference types="@webgpu/types/dist" />
import { getGlobal } from './global_util';
import { tensorToString } from './tensor_format';
import * as util from './util';
import { computeStrides, toNestedArray } from './util';
/**
 * A mutable object, similar to `tf.Tensor`, that allows users to set values
 * at locations before converting to an immutable `tf.Tensor`.
 *
 * See `tf.buffer` for creating a tensor buffer.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
export class TensorBuffer {
    constructor(shape, dtype, values) {
        this.dtype = dtype;
        this.shape = shape.slice();
        this.size = util.sizeFromShape(shape);
        if (values != null) {
            const n = values.length;
            util.assert(n === this.size, () => `Length of values '${n}' does not match the size ` +
                `inferred by the shape '${this.size}'.`);
        }
        if (dtype === 'complex64') {
            throw new Error(`complex64 dtype TensorBuffers are not supported. Please create ` +
                `a TensorBuffer for the real and imaginary parts separately and ` +
                `call tf.complex(real, imag).`);
        }
        this.values = values || util.getArrayFromDType(dtype, this.size);
        this.strides = computeStrides(shape);
    }
    /**
     * Sets a value in the buffer at a given location.
     *
     * @param value The value to set.
     * @param locs  The location indices.
     *
     * @doc {heading: 'Tensors', subheading: 'Creation'}
     */
    set(value, ...locs) {
        if (locs.length === 0) {
            locs = [0];
        }
        util.assert(locs.length === this.rank, () => `The number of provided coordinates (${locs.length}) must ` +
            `match the rank (${this.rank})`);
        const index = this.locToIndex(locs);
        this.values[index] = value;
    }
    /**
     * Returns the value in the buffer at the provided location.
     *
     * @param locs The location indices.
     *
     * @doc {heading: 'Tensors', subheading: 'Creation'}
     */
    get(...locs) {
        if (locs.length === 0) {
            locs = [0];
        }
        let i = 0;
        for (const loc of locs) {
            if (loc < 0 || loc >= this.shape[i]) {
                const msg = `Requested out of range element at ${locs}. ` +
                    `  Buffer shape=${this.shape}`;
                throw new Error(msg);
            }
            i++;
        }
        let index = locs[locs.length - 1];
        for (let i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.values[index];
    }
    locToIndex(locs) {
        if (this.rank === 0) {
            return 0;
        }
        else if (this.rank === 1) {
            return locs[0];
        }
        let index = locs[locs.length - 1];
        for (let i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return index;
    }
    indexToLoc(index) {
        if (this.rank === 0) {
            return [];
        }
        else if (this.rank === 1) {
            return [index];
        }
        const locs = new Array(this.shape.length);
        for (let i = 0; i < locs.length - 1; ++i) {
            locs[i] = Math.floor(index / this.strides[i]);
            index -= locs[i] * this.strides[i];
        }
        locs[locs.length - 1] = index;
        return locs;
    }
    get rank() {
        return this.shape.length;
    }
    /**
     * Creates an immutable `tf.Tensor` object from the buffer.
     *
     * @doc {heading: 'Tensors', subheading: 'Creation'}
     */
    toTensor() {
        return trackerFn().makeTensor(this.values, this.shape, this.dtype);
    }
}
// For tracking tensor creation and disposal.
let trackerFn = null;
// Used by chaining methods to call into ops.
let opHandler = null;
// Used to warn about deprecated methods.
let deprecationWarningFn = null;
// This here so that we can use this method on dev branches and keep the
// functionality at master.
// tslint:disable-next-line:no-unused-expression
[deprecationWarningFn];
/**
 * An external consumer can register itself as the tensor tracker. This way
 * the Tensor class can notify the tracker for every tensor created and
 * disposed.
 */
export function setTensorTracker(fn) {
    trackerFn = fn;
}
/**
 * An external consumer can register itself as the op handler. This way the
 * Tensor class can have chaining methods that call into ops via the op
 * handler.
 */
export function setOpHandler(handler) {
    opHandler = handler;
}
/**
 * Sets the deprecation warning function to be used by this file. This way the
 * Tensor class can be a leaf but still use the environment.
 */
export function setDeprecationWarningFn(fn) {
    deprecationWarningFn = fn;
}
/**
 * A `tf.Tensor` object represents an immutable, multidimensional array of
 * numbers that has a shape and a data type.
 *
 * For performance reasons, functions that create tensors do not necessarily
 * perform a copy of the data passed to them (e.g. if the data is passed as a
 * `Float32Array`), and changes to the data will change the tensor. This is not
 * a feature and is not supported. To avoid this behavior, use the tensor before
 * changing the input data or create a copy with `copy = tf.add(yourTensor, 0)`.
 *
 * See `tf.tensor` for details on how to create a `tf.Tensor`.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
export class Tensor {
    constructor(shape, dtype, dataId, id) {
        /** Whether this tensor has been globally kept. */
        this.kept = false;
        this.isDisposedInternal = false;
        this.shape = shape.slice();
        this.dtype = dtype || 'float32';
        this.size = util.sizeFromShape(shape);
        this.strides = computeStrides(shape);
        this.dataId = dataId;
        this.id = id;
        this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
    }
    get rank() {
        return this.shape.length;
    }
    /**
     * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    async buffer() {
        const vals = await this.data();
        return opHandler.buffer(this.shape, this.dtype, vals);
    }
    /**
     * Returns a `tf.TensorBuffer` that holds the underlying data.
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    bufferSync() {
        return opHandler.buffer(this.shape, this.dtype, this.dataSync());
    }
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * asynchronously.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    async array() {
        const vals = await this.data();
        return toNestedArray(this.shape, vals, this.dtype === 'complex64');
    }
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * synchronously.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    arraySync() {
        return toNestedArray(this.shape, this.dataSync(), this.dtype === 'complex64');
    }
    /**
     * Asynchronously downloads the values from the `tf.Tensor`. Returns a
     * promise of `TypedArray` that resolves when the computation has finished.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    async data() {
        this.throwIfDisposed();
        const data = trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            const bytes = await data;
            try {
                return bytes.map(b => util.decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    /**
     * Copy the tensor's data to a new GPU resource. Comparing to the `dataSync()`
     * and `data()`, this method prevents data from being downloaded to CPU.
     *
     * For WebGL backend, the data will be stored on a densely packed texture.
     * This means that the texture will use the RGBA channels to store value.
     *
     * For WebGPU backend, the data will be stored on a buffer. There is no
     * parameter, so can not use a user-defined size to create the buffer.
     *
     * @param options:
     *     For WebGL,
     *         - customTexShape: Optional. If set, will use the user defined
     *     texture shape to create the texture.
     *
     * @returns For WebGL backend, a GPUData contains the new texture and
     *     its information.
     *     {
     *        tensorRef: The tensor that is associated with this texture,
     *        texture: WebGLTexture,
     *        texShape: [number, number] // [height, width]
     *     }
     *
     *     For WebGPU backend, a GPUData contains the new buffer and
     *     its information.
     *     {
     *        tensorRef: The tensor that is associated with this buffer,
     *        buffer: GPUBuffer,
     *        bufSize: number
     *     }
     *
     *     Remember to dispose the GPUData after it is used by
     *     `res.tensorRef.dispose()`.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    dataToGPU(options) {
        this.throwIfDisposed();
        return trackerFn().readToGPU(this.dataId, options);
    }
    /**
     * Synchronously downloads the values from the `tf.Tensor`. This blocks the
     * UI thread until the values are ready, which can cause performance issues.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    dataSync() {
        this.throwIfDisposed();
        const data = trackerFn().readSync(this.dataId);
        if (this.dtype === 'string') {
            try {
                return data.map(b => util.decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    /** Returns the underlying bytes of the tensor's data. */
    async bytes() {
        this.throwIfDisposed();
        const data = await trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            return data;
        }
        else {
            return new Uint8Array(data.buffer);
        }
    }
    /**
     * Disposes `tf.Tensor` from memory.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    dispose() {
        if (this.isDisposed) {
            return;
        }
        trackerFn().disposeTensor(this);
        this.isDisposedInternal = true;
    }
    get isDisposed() {
        return this.isDisposedInternal;
    }
    throwIfDisposed() {
        if (this.isDisposed) {
            throw new Error(`Tensor is disposed.`);
        }
    }
    /**
     * Prints the `tf.Tensor`. See `tf.print` for details.
     *
     * @param verbose Whether to print verbose information about the tensor,
     *    including dtype and size.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    print(verbose = false) {
        return opHandler.print(this, verbose);
    }
    /**
     * Returns a copy of the tensor. See `tf.clone` for details.
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    clone() {
        this.throwIfDisposed();
        return opHandler.clone(this);
    }
    /**
     * Returns a human-readable description of the tensor. Useful for logging.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    toString(verbose = false) {
        const vals = this.dataSync();
        return tensorToString(vals, this.shape, this.dtype, verbose);
    }
    cast(dtype) {
        this.throwIfDisposed();
        return opHandler.cast(this, dtype);
    }
    variable(trainable = true, name, dtype) {
        this.throwIfDisposed();
        return trackerFn().makeVariable(this, trainable, name, dtype);
    }
}
Object.defineProperty(Tensor, Symbol.hasInstance, {
    value: (instance) => {
        // Implementation note: we should use properties of the object that will be
        // defined before the constructor body has finished executing (methods).
        // This is because when this code is transpiled by babel, babel will call
        // classCallCheck before the constructor body is run.
        // See https://github.com/tensorflow/tfjs/issues/3384 for backstory.
        return !!instance && instance.data != null && instance.dataSync != null &&
            instance.throwIfDisposed != null;
    }
});
export function getGlobalTensorClass() {
    // Use getGlobal so that we can augment the Tensor class across package
    // boundaries becase the node resolution alg may result in different modules
    // being returned for this file depending on the path they are loaded from.
    return getGlobal('Tensor', () => {
        return Tensor;
    });
}
// Global side effect. Cache global reference to Tensor class
getGlobalTensorClass();
/**
 * A mutable `tf.Tensor`, useful for persisting state, e.g. for training.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
export class Variable extends Tensor {
    constructor(initialValue, trainable, name, tensorId) {
        super(initialValue.shape, initialValue.dtype, initialValue.dataId, tensorId);
        this.trainable = trainable;
        this.name = name;
    }
    /**
     * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
     * the same shape and dtype as the old `tf.Tensor`.
     *
     * @param newValue New tensor to be assigned to this variable.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    assign(newValue) {
        if (newValue.dtype !== this.dtype) {
            throw new Error(`dtype of the new value (${newValue.dtype}) and ` +
                `previous value (${this.dtype}) must match`);
        }
        if (!util.arraysEqual(newValue.shape, this.shape)) {
            throw new Error(`shape of the new value (${newValue.shape}) and ` +
                `previous value (${this.shape}) must match`);
        }
        trackerFn().disposeTensor(this);
        this.dataId = newValue.dataId;
        trackerFn().incRef(this, null /* backend */);
    }
    dispose() {
        trackerFn().disposeVariable(this);
        this.isDisposedInternal = true;
    }
}
Object.defineProperty(Variable, Symbol.hasInstance, {
    value: (instance) => {
        return instance instanceof Tensor && instance.assign != null &&
            instance.assign instanceof Function;
    }
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGVuc29yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy90ZW5zb3IudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgseUVBQXlFO0FBQ3pFLDRDQUE0QztBQUU1QyxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRXhDLE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUUvQyxPQUFPLEtBQUssSUFBSSxNQUFNLFFBQVEsQ0FBQztBQUMvQixPQUFPLEVBQUMsY0FBYyxFQUFFLGFBQWEsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQVdyRDs7Ozs7OztHQU9HO0FBQ0gsTUFBTSxPQUFPLFlBQVk7SUFNdkIsWUFBWSxLQUFrQixFQUFTLEtBQVEsRUFBRSxNQUF1QjtRQUFqQyxVQUFLLEdBQUwsS0FBSyxDQUFHO1FBQzdDLElBQUksQ0FBQyxLQUFLLEdBQUcsS0FBSyxDQUFDLEtBQUssRUFBaUIsQ0FBQztRQUMxQyxJQUFJLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsS0FBSyxDQUFDLENBQUM7UUFFdEMsSUFBSSxNQUFNLElBQUksSUFBSSxFQUFFO1lBQ2xCLE1BQU0sQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLENBQUM7WUFDeEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLEtBQUssSUFBSSxDQUFDLElBQUksRUFDZixHQUFHLEVBQUUsQ0FBQyxxQkFBcUIsQ0FBQyw0QkFBNEI7Z0JBQ3BELDBCQUEwQixJQUFJLENBQUMsSUFBSSxJQUFJLENBQUMsQ0FBQztTQUNsRDtRQUNELElBQUksS0FBSyxLQUFLLFdBQVcsRUFBRTtZQUN6QixNQUFNLElBQUksS0FBSyxDQUNYLGlFQUFpRTtnQkFDakUsaUVBQWlFO2dCQUNqRSw4QkFBOEIsQ0FBQyxDQUFDO1NBQ3JDO1FBQ0QsSUFBSSxDQUFDLE1BQU0sR0FB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QVEsQ0FBQztJQUMxQyxDQUFDO0NBQ0YsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vLyBXb3JrYXJvdW5kIGZvcjogaHR0cHM6Ly9naXRodWIuY29tL2JhemVsYnVpbGQvcnVsZXNfbm9kZWpzL2lzc3Vlcy8xMjY1XG4vLy8gPHJlZmVyZW5jZSB0eXBlcz1cIkB3ZWJncHUvdHlwZXMvZGlzdFwiIC8+XG5cbmltcG9ydCB7Z2V0R2xvYmFsfSBmcm9tICcuL2dsb2JhbF91dGlsJztcbmltcG9ydCB7VGVuc29ySW5mbywgRGF0YUlkfSBmcm9tICcuL3RlbnNvcl9pbmZvJztcbmltcG9ydCB7dGVuc29yVG9TdHJpbmd9IGZyb20gJy4vdGVuc29yX2Zvcm1hdCc7XG5pbXBvcnQge0FycmF5TWFwLCBCYWNrZW5kVmFsdWVzLCBEYXRhVHlwZSwgRGF0YVR5cGVNYXAsIERhdGFWYWx1ZXMsIE51bWVyaWNEYXRhVHlwZSwgUmFuaywgU2hhcGVNYXAsIFNpbmdsZVZhbHVlTWFwLCBUeXBlZEFycmF5fSBmcm9tICcuL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi91dGlsJztcbmltcG9ydCB7Y29tcHV0ZVN0cmlkZXMsIHRvTmVzdGVkQXJyYXl9IGZyb20gJy4vdXRpbCc7XG5cbmV4cG9ydCBpbnRlcmZhY2UgVGVuc29yRGF0YTxEIGV4dGVuZHMgRGF0YVR5cGU+IHtcbiAgZGF0YUlkPzogRGF0YUlkO1xuICB2YWx1ZXM/OiBEYXRhVHlwZU1hcFtEXTtcbn1cblxuLy8gVGhpcyBpbnRlcmZhY2UgbWltaWNzIEtlcm5lbEJhY2tlbmQgKGluIGJhY2tlbmQudHMpLCB3aGljaCB3b3VsZCBjcmVhdGUgYVxuLy8gY2lyY3VsYXIgZGVwZW5kZW5jeSBpZiBpbXBvcnRlZC5cbmV4cG9ydCBpbnRlcmZhY2UgQmFja2VuZCB7fVxuXG4vKipcbiAqIEEgbXV0YWJsZSBvYmplY3QsIHNpbWlsYXIgdG8gYHRmLlRlbnNvcmAsIHRoYXQgYWxsb3dzIHVzZXJzIHRvIHNldCB2YWx1ZXNcbiAqIGF0IGxvY2F0aW9ucyBiZWZvcmUgY29udmVydGluZyB0byBhbiBpbW11dGFibGUgYHRmLlRlbnNvcmAuXG4gKlxuICogU2VlIGB0Zi5idWZmZXJgIGZvciBjcmVhdGluZyBhIHRlbnNvciBidWZmZXIuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gKi9cbmV4cG9ydCBjbGFzcyBUZW5zb3JCdWZmZXI8UiBleHRlbmRzIFJhbmssIEQgZXh0ZW5kcyBEYXRhVHlwZSA9ICdmbG9hdDMyJz4ge1xuICBzaXplOiBudW1iZXI7XG4gIHNoYXBlOiBTaGFwZU1hcFtSXTtcbiAgc3RyaWRlczogbnVtYmVyW107XG4gIHZhbHVlczogRGF0YVR5cGVNYXBbRF07XG5cbiAgY29uc3RydWN0b3Ioc2hhcGU6IFNoYXBlTWFwW1JdLCBwdWJsaWMgZHR5cGU6IEQsIHZhbHVlcz86IERhdGFUeXBlTWFwW0RdKSB7XG4gICAgdGhpcy5zaGFwZSA9IHNoYXBlLnNsaWNlKCkgYXMgU2hhcGVNYXBbUl07XG4gICAgdGhpcy5zaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcblxuICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgY29uc3QgbiA9IHZhbHVlcy5sZW5ndGg7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBuID09PSB0aGlzLnNpemUsXG4gICAgICAgICAgKCkgPT4gYExlbmd0aCBvZiB2YWx1ZXMgJyR7bn0nIGRvZXMgbm90IG1hdGNoIHRoZSBzaXplIGAgK1xuICAgICAgICAgICAgICBgaW5mZXJyZWQgYnkgdGhlIHNoYXBlICcke3RoaXMuc2l6ZX0nLmApO1xuICAgIH1cbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYGNvbXBsZXg2NCBkdHlwZSBUZW5zb3JCdWZmZXJzIGFyZSBub3Qgc3VwcG9ydGVkLiBQbGVhc2UgY3JlYXRlIGAgK1xuICAgICAgICAgIGBhIFRlbnNvckJ1ZmZlciBmb3IgdGhlIHJlYWwgYW5kIGltYWdpbmFyeSBwYXJ0cyBzZXBhcmF0ZWx5IGFuZCBgICtcbiAgICAgICAgICBgY2FsbCB0Zi5jb21wbGV4KHJlYWwsIGltYWcpLmApO1xuICAgIH1cbiAgICB0aGlzLnZhbHVlcyA9IHZhbHVlcyB8fCB1dGlsLmdldEFycmF5RnJvbURUeXBlKGR0eXBlLCB0aGlzLnNpemUpO1xuICAgIHRoaXMuc3RyaWRlcyA9IGNvbXB1dGVTdHJpZGVzKHNoYXBlKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTZXRzIGEgdmFsdWUgaW4gdGhlIGJ1ZmZlciBhdCBhIGdpdmVuIGxvY2F0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0gdmFsdWUgVGhlIHZhbHVlIHRvIHNldC5cbiAgICogQHBhcmFtIGxvY3MgIFRoZSBsb2NhdGlvbiBpbmRpY2VzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDcmVhdGlvbid9XG4gICAqL1xuICBzZXQodmFsdWU6IFNpbmdsZVZhbHVlTWFwW0RdLCAuLi5sb2NzOiBudW1iZXJbXSk6IHZvaWQge1xuICAgIGlmIChsb2NzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgbG9jcyA9IFswXTtcbiAgICB9XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGxvY3MubGVuZ3RoID09PSB0aGlzLnJhbmssXG4gICAgICAgICgpID0+IGBUaGUgbnVtYmVyIG9mIHByb3ZpZGVkIGNvb3JkaW5hdGVzICgke2xvY3MubGVuZ3RofSkgbXVzdCBgICtcbiAgICAgICAgICAgIGBtYXRjaCB0aGUgcmFuayAoJHt0aGlzLnJhbmt9KWApO1xuXG4gICAgY29uc3QgaW5kZXggPSB0aGlzLmxvY1RvSW5kZXgobG9jcyk7XG4gICAgdGhpcy52YWx1ZXNbaW5kZXhdID0gdmFsdWUgYXMgbnVtYmVyO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgdGhlIHZhbHVlIGluIHRoZSBidWZmZXIgYXQgdGhlIHByb3ZpZGVkIGxvY2F0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0gbG9jcyBUaGUgbG9jYXRpb24gaW5kaWNlcy5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICAgKi9cbiAgZ2V0KC4uLmxvY3M6IG51bWJlcltdKTogU2luZ2xlVmFsdWVNYXBbRF0ge1xuICAgIGlmIChsb2NzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgbG9jcyA9IFswXTtcbiAgICB9XG4gICAgbGV0IGkgPSAwO1xuICAgIGZvciAoY29uc3QgbG9jIG9mIGxvY3MpIHtcbiAgICAgIGlmIChsb2MgPCAwIHx8IGxvYyA+PSB0aGlzLnNoYXBlW2ldKSB7XG4gICAgICAgIGNvbnN0IG1zZyA9IGBSZXF1ZXN0ZWQgb3V0IG9mIHJhbmdlIGVsZW1lbnQgYXQgJHtsb2NzfS4gYCArXG4gICAgICAgICAgICBgICBCdWZmZXIgc2hhcGU9JHt0aGlzLnNoYXBlfWA7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihtc2cpO1xuICAgICAgfVxuICAgICAgaSsrO1xuICAgIH1cbiAgICBsZXQgaW5kZXggPSBsb2NzW2xvY3MubGVuZ3RoIC0gMV07XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgaW5kZXggKz0gdGhpcy5zdHJpZGVzW2ldICogbG9jc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMudmFsdWVzW2luZGV4XSBhcyBTaW5nbGVWYWx1ZU1hcFtEXTtcbiAgfVxuXG4gIGxvY1RvSW5kZXgobG9jczogbnVtYmVyW10pOiBudW1iZXIge1xuICAgIGlmICh0aGlzLnJhbmsgPT09IDApIHtcbiAgICAgIHJldHVybiAwO1xuICAgIH0gZWxzZSBpZiAodGhpcy5yYW5rID09PSAxKSB7XG4gICAgICByZXR1cm4gbG9jc1swXTtcbiAgICB9XG4gICAgbGV0IGluZGV4ID0gbG9jc1tsb2NzLmxlbmd0aCAtIDFdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICAgIGluZGV4ICs9IHRoaXMuc3RyaWRlc1tpXSAqIGxvY3NbaV07XG4gICAgfVxuICAgIHJldHVybiBpbmRleDtcbiAgfVxuXG4gIGluZGV4VG9Mb2MoaW5kZXg6IG51bWJlcik6IG51bWJlcltdIHtcbiAgICBpZiAodGhpcy5yYW5rID09PSAwKSB7XG4gICAgICByZXR1cm4gW107XG4gICAgfSBlbHNlIGlmICh0aGlzLnJhbmsgPT09IDEpIHtcbiAgICAgIHJldHVybiBbaW5kZXhdO1xuICAgIH1cbiAgICBjb25zdCBsb2NzOiBudW1iZXJbXSA9IG5ldyBBcnJheSh0aGlzLnNoYXBlLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgbG9jc1tpXSA9IE1hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZXNbaV0pO1xuICAgICAgaW5kZXggLT0gbG9jc1tpXSAqIHRoaXMuc3RyaWRlc1tpXTtcbiAgICB9XG4gICAgbG9jc1tsb2NzLmxlbmd0aCAtIDFdID0gaW5kZXg7XG4gICAgcmV0dXJuIGxvY3M7XG4gIH1cblxuICBnZXQgcmFuaygpIHtcbiAgICByZXR1cm4gdGhpcy5zaGFwZS5sZW5ndGg7XG4gIH1cblxuICAvKipcbiAgICogQ3JlYXRlcyBhbiBpbW11dGFibGUgYHRmLlRlbnNvcmAgb2JqZWN0IGZyb20gdGhlIGJ1ZmZlci5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICAgKi9cbiAgdG9UZW5zb3IoKTogVGVuc29yPFI+IHtcbiAgICByZXR1cm4gdHJhY2tlckZuKCkubWFrZVRlbnNvcih0aGlzLnZhbHVlcywgdGhpcy5zaGFwZSwgdGhpcy5kdHlwZSkgYXNcbiAgICAgICAgVGVuc29yPFI+O1xuICB9XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgRGF0YVRvR1BVV2ViR0xPcHRpb24ge1xuICBjdXN0b21UZXhTaGFwZT86IFtudW1iZXIsIG51bWJlcl07XG59XG5cbmV4cG9ydCB0eXBlIERhdGFUb0dQVU9wdGlvbnMgPSBEYXRhVG9HUFVXZWJHTE9wdGlvbjtcblxuZXhwb3J0IGludGVyZmFjZSBHUFVEYXRhIHtcbiAgdGVuc29yUmVmOiBUZW5zb3I7XG4gIHRleHR1cmU/OiBXZWJHTFRleHR1cmU7XG4gIGJ1ZmZlcj86IEdQVUJ1ZmZlcjtcbiAgdGV4U2hhcGU/OiBbbnVtYmVyLCBudW1iZXJdO1xuICBidWZTaXplPzogbnVtYmVyO1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIFRlbnNvclRyYWNrZXIge1xuICBtYWtlVGVuc29yKFxuICAgICAgdmFsdWVzOiBEYXRhVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSxcbiAgICAgIGJhY2tlbmQ/OiBCYWNrZW5kKTogVGVuc29yO1xuICBtYWtlVmFyaWFibGUoXG4gICAgICBpbml0aWFsVmFsdWU6IFRlbnNvciwgdHJhaW5hYmxlPzogYm9vbGVhbiwgbmFtZT86IHN0cmluZyxcbiAgICAgIGR0eXBlPzogRGF0YVR5cGUpOiBWYXJpYWJsZTtcbiAgaW5jUmVmKGE6IFRlbnNvciwgYmFja2VuZDogQmFja2VuZCk6IHZvaWQ7XG4gIGRpc3Bvc2VUZW5zb3IodDogVGVuc29yKTogdm9pZDtcbiAgZGlzcG9zZVZhcmlhYmxlKHY6IFZhcmlhYmxlKTogdm9pZDtcbiAgcmVhZChkYXRhSWQ6IERhdGFJZCk6IFByb21pc2U8QmFja2VuZFZhbHVlcz47XG4gIHJlYWRTeW5jKGRhdGFJZDogRGF0YUlkKTogQmFja2VuZFZhbHVlcztcbiAgcmVhZFRvR1BVKGRhdGFJZDogRGF0YUlkLCBvcHRpb25zPzogRGF0YVRvR1BVT3B0aW9ucyk6IEdQVURhdGE7XG59XG5cbi8qKlxuICogVGhlIFRlbnNvciBjbGFzcyBjYWxscyBpbnRvIHRoaXMgaGFuZGxlciB0byBkZWxlZ2F0ZSBjaGFpbmluZyBvcGVyYXRpb25zLlxuICovXG5leHBvcnQgaW50ZXJmYWNlIE9wSGFuZGxlciB7XG4gIGNhc3Q8VCBleHRlbmRzIFRlbnNvcj4oeDogVCwgZHR5cGU6IERhdGFUeXBlKTogVDtcbiAgYnVmZmVyPFIgZXh0ZW5kcyBSYW5rLCBEIGV4dGVuZHMgRGF0YVR5cGU+KFxuICAgICAgc2hhcGU6IFNoYXBlTWFwW1JdLCBkdHlwZTogRCxcbiAgICAgIHZhbHVlcz86IERhdGFUeXBlTWFwW0RdKTogVGVuc29yQnVmZmVyPFIsIEQ+O1xuICBwcmludDxUIGV4dGVuZHMgVGVuc29yPih4OiBULCB2ZXJib3NlOiBib29sZWFuKTogdm9pZDtcbiAgY2xvbmU8VCBleHRlbmRzIFRlbnNvcj4oeDogVCk6IFQ7XG4gIC8vIFRPRE8oeWFzc29nYmEpIGJyaW5nIHJlc2hhcGUgYmFjaz9cbn1cblxuLy8gRm9yIHRyYWNraW5nIHRlbnNvciBjcmVhdGlvbiBhbmQgZGlzcG9zYWwuXG5sZXQgdHJhY2tlckZuOiAoKSA9PiBUZW5zb3JUcmFja2VyID0gbnVsbDtcbi8vIFVzZWQgYnkgY2hhaW5pbmcgbWV0aG9kcyB0byBjYWxsIGludG8gb3BzLlxubGV0IG9wSGFuZGxlcjogT3BIYW5kbGVyID0gbnVsbDtcbi8vIFVzZWQgdG8gd2FybiBhYm91dCBkZXByZWNhdGVkIG1ldGhvZHMuXG5sZXQgZGVwcmVjYXRpb25XYXJuaW5nRm46IChtc2c6IHN0cmluZykgPT4gdm9pZCA9IG51bGw7XG4vLyBUaGlzIGhlcmUgc28gdGhhdCB3ZSBjYW4gdXNlIHRoaXMgbWV0aG9kIG9uIGRldiBicmFuY2hlcyBhbmQga2VlcCB0aGVcbi8vIGZ1bmN0aW9uYWxpdHkgYXQgbWFzdGVyLlxuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLXVudXNlZC1leHByZXNzaW9uXG5bZGVwcmVjYXRpb25XYXJuaW5nRm5dO1xuXG4vKipcbiAqIEFuIGV4dGVybmFsIGNvbnN1bWVyIGNhbiByZWdpc3RlciBpdHNlbGYgYXMgdGhlIHRlbnNvciB0cmFja2VyLiBUaGlzIHdheVxuICogdGhlIFRlbnNvciBjbGFzcyBjYW4gbm90aWZ5IHRoZSB0cmFja2VyIGZvciBldmVyeSB0ZW5zb3IgY3JlYXRlZCBhbmRcbiAqIGRpc3Bvc2VkLlxuICovXG5leHBvcnQgZnVuY3Rpb24gc2V0VGVuc29yVHJhY2tlcihmbjogKCkgPT4gVGVuc29yVHJhY2tlcikge1xuICB0cmFja2VyRm4gPSBmbjtcbn1cblxuLyoqXG4gKiBBbiBleHRlcm5hbCBjb25zdW1lciBjYW4gcmVnaXN0ZXIgaXRzZWxmIGFzIHRoZSBvcCBoYW5kbGVyLiBUaGlzIHdheSB0aGVcbiAqIFRlbnNvciBjbGFzcyBjYW4gaGF2ZSBjaGFpbmluZyBtZXRob2RzIHRoYXQgY2FsbCBpbnRvIG9wcyB2aWEgdGhlIG9wXG4gKiBoYW5kbGVyLlxuICovXG5leHBvcnQgZnVuY3Rpb24gc2V0T3BIYW5kbGVyKGhhbmRsZXI6IE9wSGFuZGxlcikge1xuICBvcEhhbmRsZXIgPSBoYW5kbGVyO1xufVxuXG4vKipcbiAqIFNldHMgdGhlIGRlcHJlY2F0aW9uIHdhcm5pbmcgZnVuY3Rpb24gdG8gYmUgdXNlZCBieSB0aGlzIGZpbGUuIFRoaXMgd2F5IHRoZVxuICogVGVuc29yIGNsYXNzIGNhbiBiZSBhIGxlYWYgYnV0IHN0aWxsIHVzZSB0aGUgZW52aXJvbm1lbnQuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzZXREZXByZWNhdGlvbldhcm5pbmdGbihmbjogKG1zZzogc3RyaW5nKSA9PiB2b2lkKSB7XG4gIGRlcHJlY2F0aW9uV2FybmluZ0ZuID0gZm47XG59XG5cbi8vIERlY2xhcmUgdGhpcyBuYW1lc3BhY2UgdG8gbWFrZSBUZW5zb3IgY2xhc3MgYXVnbWVudGF0aW9uIHdvcmsgaW4gZ29vZ2xlMy5cbmV4cG9ydCBkZWNsYXJlIG5hbWVzcGFjZSBUZW5zb3Ige31cbi8qKlxuICogQSBgdGYuVGVuc29yYCBvYmplY3QgcmVwcmVzZW50cyBhbiBpbW11dGFibGUsIG11bHRpZGltZW5zaW9uYWwgYXJyYXkgb2ZcbiAqIG51bWJlcnMgdGhhdCBoYXMgYSBzaGFwZSBhbmQgYSBkYXRhIHR5cGUuXG4gKlxuICogRm9yIHBlcmZvcm1hbmNlIHJlYXNvbnMsIGZ1bmN0aW9ucyB0aGF0IGNyZWF0ZSB0ZW5zb3JzIGRvIG5vdCBuZWNlc3NhcmlseVxuICogcGVyZm9ybSBhIGNvcHkgb2YgdGhlIGRhdGEgcGFzc2VkIHRvIHRoZW0gKGUuZy4gaWYgdGhlIGRhdGEgaXMgcGFzc2VkIGFzIGFcbiAqIGBGbG9hdDMyQXJyYXlgKSwgYW5kIGNoYW5nZXMgdG8gdGhlIGRhdGEgd2lsbCBjaGFuZ2UgdGhlIHRlbnNvci4gVGhpcyBpcyBub3RcbiAqIGEgZmVhdHVyZSBhbmQgaXMgbm90IHN1cHBvcnRlZC4gVG8gYXZvaWQgdGhpcyBiZWhhdmlvciwgdXNlIHRoZSB0ZW5zb3IgYmVmb3JlXG4gKiBjaGFuZ2luZyB0aGUgaW5wdXQgZGF0YSBvciBjcmVhdGUgYSBjb3B5IHdpdGggYGNvcHkgPSB0Zi5hZGQoeW91clRlbnNvciwgMClgLlxuICpcbiAqIFNlZSBgdGYudGVuc29yYCBmb3IgZGV0YWlscyBvbiBob3cgdG8gY3JlYXRlIGEgYHRmLlRlbnNvcmAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gKi9cbmV4cG9ydCBjbGFzcyBUZW5zb3I8UiBleHRlbmRzIFJhbmsgPSBSYW5rPiBpbXBsZW1lbnRzIFRlbnNvckluZm8ge1xuICAvKiogVW5pcXVlIGlkIG9mIHRoaXMgdGVuc29yLiAqL1xuICByZWFkb25seSBpZDogbnVtYmVyO1xuICAvKipcbiAgICogSWQgb2YgdGhlIGJ1Y2tldCBob2xkaW5nIHRoZSBkYXRhIGZvciB0aGlzIHRlbnNvci4gTXVsdGlwbGUgYXJyYXlzIGNhblxuICAgKiBwb2ludCB0byB0aGUgc2FtZSBidWNrZXQgKGUuZy4gd2hlbiBjYWxsaW5nIGFycmF5LnJlc2hhcGUoKSkuXG4gICAqL1xuICBkYXRhSWQ6IERhdGFJZDtcbiAgLyoqIFRoZSBzaGFwZSBvZiB0aGUgdGVuc29yLiAqL1xuICByZWFkb25seSBzaGFwZTogU2hhcGVNYXBbUl07XG4gIC8qKiBOdW1iZXIgb2YgZWxlbWVudHMgaW4gdGhlIHRlbnNvci4gKi9cbiAgcmVhZG9ubHkgc2l6ZTogbnVtYmVyO1xuICAvKiogVGhlIGRhdGEgdHlwZSBmb3IgdGhlIGFycmF5LiAqL1xuICByZWFkb25seSBkdHlwZTogRGF0YVR5cGU7XG4gIC8qKiBUaGUgcmFuayB0eXBlIGZvciB0aGUgYXJyYXkgKHNlZSBgUmFua2AgZW51bSkuICovXG4gIHJlYWRvbmx5IHJhbmtUeXBlOiBSO1xuXG4gIC8qKiBXaGV0aGVyIHRoaXMgdGVuc29yIGhhcyBiZWVuIGdsb2JhbGx5IGtlcHQuICovXG4gIGtlcHQgPSBmYWxzZTtcbiAgLyoqIFRoZSBpZCBvZiB0aGUgc2NvcGUgdGhpcyB0ZW5zb3IgaXMgYmVpbmcgdHJhY2tlZCBpbi4gKi9cbiAgc2NvcGVJZDogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBOdW1iZXIgb2YgZWxlbWVudHMgdG8gc2tpcCBpbiBlYWNoIGRpbWVuc2lvbiB3aGVuIGluZGV4aW5nLiBTZWVcbiAgICogaHR0cHM6Ly9kb2NzLnNjaXB5Lm9yZy9kb2MvbnVtcHkvcmVmZXJlbmNlL2dlbmVyYXRlZC9cXFxuICAgKiBudW1weS5uZGFycmF5LnN0cmlkZXMuaHRtbFxuICAgKi9cbiAgcmVhZG9ubHkgc3RyaWRlczogbnVtYmVyW107XG5cbiAgY29uc3RydWN0b3Ioc2hhcGU6IFNoYXBlTWFwW1JdLCBkdHlwZTogRGF0YVR5cGUsIGRhdGFJZDogRGF0YUlkLCBpZDogbnVtYmVyKSB7XG4gICAgdGhpcy5zaGFwZSA9IHNoYXBlLnNsaWNlKCkgYXMgU2hhcGVNYXBbUl07XG4gICAgdGhpcy5kdHlwZSA9IGR0eXBlIHx8ICdmbG9hdDMyJztcbiAgICB0aGlzLnNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgIHRoaXMuc3RyaWRlcyA9IGNvbXB1dGVTdHJpZGVzKHNoYXBlKTtcbiAgICB0aGlzLmRhdGFJZCA9IGRhdGFJZDtcbiAgICB0aGlzLmlkID0gaWQ7XG4gICAgdGhpcy5yYW5rVHlwZSA9ICh0aGlzLnJhbmsgPCA1ID8gdGhpcy5yYW5rLnRvU3RyaW5nKCkgOiAnaGlnaGVyJykgYXMgUjtcbiAgfVxuXG4gIGdldCByYW5rKCk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuc2hhcGUubGVuZ3RoO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgYSBwcm9taXNlIG9mIGB0Zi5UZW5zb3JCdWZmZXJgIHRoYXQgaG9sZHMgdGhlIHVuZGVybHlpbmcgZGF0YS5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBhc3luYyBidWZmZXI8RCBleHRlbmRzIERhdGFUeXBlID0gJ2Zsb2F0MzInPigpOiBQcm9taXNlPFRlbnNvckJ1ZmZlcjxSLCBEPj4ge1xuICAgIGNvbnN0IHZhbHMgPSBhd2FpdCB0aGlzLmRhdGE8RD4oKTtcbiAgICByZXR1cm4gb3BIYW5kbGVyLmJ1ZmZlcih0aGlzLnNoYXBlLCB0aGlzLmR0eXBlIGFzIEQsIHZhbHMpO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgYSBgdGYuVGVuc29yQnVmZmVyYCB0aGF0IGhvbGRzIHRoZSB1bmRlcmx5aW5nIGRhdGEuXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgYnVmZmVyU3luYzxEIGV4dGVuZHMgRGF0YVR5cGUgPSAnZmxvYXQzMic+KCk6IFRlbnNvckJ1ZmZlcjxSLCBEPiB7XG4gICAgcmV0dXJuIG9wSGFuZGxlci5idWZmZXIodGhpcy5zaGFwZSwgdGhpcy5kdHlwZSBhcyBELCB0aGlzLmRhdGFTeW5jKCkpO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgdGhlIHRlbnNvciBkYXRhIGFzIGEgbmVzdGVkIGFycmF5LiBUaGUgdHJhbnNmZXIgb2YgZGF0YSBpcyBkb25lXG4gICAqIGFzeW5jaHJvbm91c2x5LlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFzeW5jIGFycmF5KCk6IFByb21pc2U8QXJyYXlNYXBbUl0+IHtcbiAgICBjb25zdCB2YWxzID0gYXdhaXQgdGhpcy5kYXRhKCk7XG4gICAgcmV0dXJuIHRvTmVzdGVkQXJyYXkodGhpcy5zaGFwZSwgdmFscywgdGhpcy5kdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIGFzXG4gICAgICAgIEFycmF5TWFwW1JdO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgdGhlIHRlbnNvciBkYXRhIGFzIGEgbmVzdGVkIGFycmF5LiBUaGUgdHJhbnNmZXIgb2YgZGF0YSBpcyBkb25lXG4gICAqIHN5bmNocm9ub3VzbHkuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgYXJyYXlTeW5jKCk6IEFycmF5TWFwW1JdIHtcbiAgICByZXR1cm4gdG9OZXN0ZWRBcnJheShcbiAgICAgICAgICAgICAgIHRoaXMuc2hhcGUsIHRoaXMuZGF0YVN5bmMoKSwgdGhpcy5kdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIGFzXG4gICAgICAgIEFycmF5TWFwW1JdO1xuICB9XG5cbiAgLyoqXG4gICAqIEFzeW5jaHJvbm91c2x5IGRvd25sb2FkcyB0aGUgdmFsdWVzIGZyb20gdGhlIGB0Zi5UZW5zb3JgLiBSZXR1cm5zIGFcbiAgICogcHJvbWlzZSBvZiBgVHlwZWRBcnJheWAgdGhhdCByZXNvbHZlcyB3aGVuIHRoZSBjb21wdXRhdGlvbiBoYXMgZmluaXNoZWQuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgYXN5bmMgZGF0YTxEIGV4dGVuZHMgRGF0YVR5cGUgPSBOdW1lcmljRGF0YVR5cGU+KCk6IFByb21pc2U8RGF0YVR5cGVNYXBbRF0+IHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IGRhdGEgPSB0cmFja2VyRm4oKS5yZWFkKHRoaXMuZGF0YUlkKTtcbiAgICBpZiAodGhpcy5kdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIGNvbnN0IGJ5dGVzID0gYXdhaXQgZGF0YSBhcyBVaW50OEFycmF5W107XG4gICAgICB0cnkge1xuICAgICAgICByZXR1cm4gYnl0ZXMubWFwKGIgPT4gdXRpbC5kZWNvZGVTdHJpbmcoYikpIGFzIERhdGFUeXBlTWFwW0RdO1xuICAgICAgfSBjYXRjaCB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdGYWlsZWQgdG8gZGVjb2RlIHRoZSBzdHJpbmcgYnl0ZXMgaW50byB1dGYtOC4gJyArXG4gICAgICAgICAgICAnVG8gZ2V0IHRoZSBvcmlnaW5hbCBieXRlcywgY2FsbCB0ZW5zb3IuYnl0ZXMoKS4nKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGRhdGEgYXMgUHJvbWlzZTxEYXRhVHlwZU1hcFtEXT47XG4gIH1cblxuICAvKipcbiAgICogQ29weSB0aGUgdGVuc29yJ3MgZGF0YSB0byBhIG5ldyBHUFUgcmVzb3VyY2UuIENvbXBhcmluZyB0byB0aGUgYGRhdGFTeW5jKClgXG4gICAqIGFuZCBgZGF0YSgpYCwgdGhpcyBtZXRob2QgcHJldmVudHMgZGF0YSBmcm9tIGJlaW5nIGRvd25sb2FkZWQgdG8gQ1BVLlxuICAgKlxuICAgKiBGb3IgV2ViR0wgYmFja2VuZCwgdGhlIGRhdGEgd2lsbCBiZSBzdG9yZWQgb24gYSBkZW5zZWx5IHBhY2tlZCB0ZXh0dXJlLlxuICAgKiBUaGlzIG1lYW5zIHRoYXQgdGhlIHRleHR1cmUgd2lsbCB1c2UgdGhlIFJHQkEgY2hhbm5lbHMgdG8gc3RvcmUgdmFsdWUuXG4gICAqXG4gICAqIEZvciBXZWJHUFUgYmFja2VuZCwgdGhlIGRhdGEgd2lsbCBiZSBzdG9yZWQgb24gYSBidWZmZXIuIFRoZXJlIGlzIG5vXG4gICAqIHBhcmFtZXRlciwgc28gY2FuIG5vdCB1c2UgYSB1c2VyLWRlZmluZWQgc2l6ZSB0byBjcmVhdGUgdGhlIGJ1ZmZlci5cbiAgICpcbiAgICogQHBhcmFtIG9wdGlvbnM6XG4gICAqICAgICBGb3IgV2ViR0wsXG4gICAqICAgICAgICAgLSBjdXN0b21UZXhTaGFwZTogT3B0aW9uYWwuIElmIHNldCwgd2lsbCB1c2UgdGhlIHVzZXIgZGVmaW5lZFxuICAgKiAgICAgdGV4dHVyZSBzaGFwZSB0byBjcmVhdGUgdGhlIHRleHR1cmUuXG4gICAqXG4gICAqIEByZXR1cm5zIEZvciBXZWJHTCBiYWNrZW5kLCBhIEdQVURhdGEgY29udGFpbnMgdGhlIG5ldyB0ZXh0dXJlIGFuZFxuICAgKiAgICAgaXRzIGluZm9ybWF0aW9uLlxuICAgKiAgICAge1xuICAgKiAgICAgICAgdGVuc29yUmVmOiBUaGUgdGVuc29yIHRoYXQgaXMgYXNzb2NpYXRlZCB3aXRoIHRoaXMgdGV4dHVyZSxcbiAgICogICAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSxcbiAgICogICAgICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdIC8vIFtoZWlnaHQsIHdpZHRoXVxuICAgKiAgICAgfVxuICAgKlxuICAgKiAgICAgRm9yIFdlYkdQVSBiYWNrZW5kLCBhIEdQVURhdGEgY29udGFpbnMgdGhlIG5ldyBidWZmZXIgYW5kXG4gICAqICAgICBpdHMgaW5mb3JtYXRpb24uXG4gICAqICAgICB7XG4gICAqICAgICAgICB0ZW5zb3JSZWY6IFRoZSB0ZW5zb3IgdGhhdCBpcyBhc3NvY2lhdGVkIHdpdGggdGhpcyBidWZmZXIsXG4gICAqICAgICAgICBidWZmZXI6IEdQVUJ1ZmZlcixcbiAgICogICAgICAgIGJ1ZlNpemU6IG51bWJlclxuICAgKiAgICAgfVxuICAgKlxuICAgKiAgICAgUmVtZW1iZXIgdG8gZGlzcG9zZSB0aGUgR1BVRGF0YSBhZnRlciBpdCBpcyB1c2VkIGJ5XG4gICAqICAgICBgcmVzLnRlbnNvclJlZi5kaXNwb3NlKClgLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGRhdGFUb0dQVShvcHRpb25zPzogRGF0YVRvR1BVT3B0aW9ucyk6IEdQVURhdGEge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIHRyYWNrZXJGbigpLnJlYWRUb0dQVSh0aGlzLmRhdGFJZCwgb3B0aW9ucyk7XG4gIH1cblxuICAvKipcbiAgICogU3luY2hyb25vdXNseSBkb3dubG9hZHMgdGhlIHZhbHVlcyBmcm9tIHRoZSBgdGYuVGVuc29yYC4gVGhpcyBibG9ja3MgdGhlXG4gICAqIFVJIHRocmVhZCB1bnRpbCB0aGUgdmFsdWVzIGFyZSByZWFkeSwgd2hpY2ggY2FuIGNhdXNlIHBlcmZvcm1hbmNlIGlzc3Vlcy5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBkYXRhU3luYzxEIGV4dGVuZHMgRGF0YVR5cGUgPSBOdW1lcmljRGF0YVR5cGU+KCk6IERhdGFUeXBlTWFwW0RdIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IGRhdGEgPSB0cmFja2VyRm4oKS5yZWFkU3luYyh0aGlzLmRhdGFJZCk7XG4gICAgaWYgKHRoaXMuZHR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgICB0cnkge1xuICAgICAgICByZXR1cm4gKGRhdGEgYXMgVWludDhBcnJheVtdKS5tYXAoYiA9PiB1dGlsLmRlY29kZVN0cmluZyhiKSkgYXNcbiAgICAgICAgICAgIERhdGFUeXBlTWFwW0RdO1xuICAgICAgfSBjYXRjaCB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdGYWlsZWQgdG8gZGVjb2RlIHRoZSBzdHJpbmcgYnl0ZXMgaW50byB1dGYtOC4gJyArXG4gICAgICAgICAgICAnVG8gZ2V0IHRoZSBvcmlnaW5hbCBieXRlcywgY2FsbCB0ZW5zb3IuYnl0ZXMoKS4nKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGRhdGEgYXMgRGF0YVR5cGVNYXBbRF07XG4gIH1cblxuICAvKiogUmV0dXJucyB0aGUgdW5kZXJseWluZyBieXRlcyBvZiB0aGUgdGVuc29yJ3MgZGF0YS4gKi9cbiAgYXN5bmMgYnl0ZXMoKTogUHJvbWlzZTxVaW50OEFycmF5W118VWludDhBcnJheT4ge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgZGF0YSA9IGF3YWl0IHRyYWNrZXJGbigpLnJlYWQodGhpcy5kYXRhSWQpO1xuICAgIGlmICh0aGlzLmR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgcmV0dXJuIGRhdGEgYXMgVWludDhBcnJheVtdO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gbmV3IFVpbnQ4QXJyYXkoKGRhdGEgYXMgVHlwZWRBcnJheSkuYnVmZmVyKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogRGlzcG9zZXMgYHRmLlRlbnNvcmAgZnJvbSBtZW1vcnkuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgZGlzcG9zZSgpOiB2b2lkIHtcbiAgICBpZiAodGhpcy5pc0Rpc3Bvc2VkKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIHRyYWNrZXJGbigpLmRpc3Bvc2VUZW5zb3IodGhpcyk7XG4gICAgdGhpcy5pc0Rpc3Bvc2VkSW50ZXJuYWwgPSB0cnVlO1xuICB9XG5cbiAgcHJvdGVjdGVkIGlzRGlzcG9zZWRJbnRlcm5hbCA9IGZhbHNlO1xuICBnZXQgaXNEaXNwb3NlZCgpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdGhpcy5pc0Rpc3Bvc2VkSW50ZXJuYWw7XG4gIH1cblxuICB0aHJvd0lmRGlzcG9zZWQoKSB7XG4gICAgaWYgKHRoaXMuaXNEaXNwb3NlZCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBUZW5zb3IgaXMgZGlzcG9zZWQuYCk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIFByaW50cyB0aGUgYHRmLlRlbnNvcmAuIFNlZSBgdGYucHJpbnRgIGZvciBkZXRhaWxzLlxuICAgKlxuICAgKiBAcGFyYW0gdmVyYm9zZSBXaGV0aGVyIHRvIHByaW50IHZlcmJvc2UgaW5mb3JtYXRpb24gYWJvdXQgdGhlIHRlbnNvcixcbiAgICogICAgaW5jbHVkaW5nIGR0eXBlIGFuZCBzaXplLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHByaW50KHZlcmJvc2UgPSBmYWxzZSk6IHZvaWQge1xuICAgIHJldHVybiBvcEhhbmRsZXIucHJpbnQodGhpcywgdmVyYm9zZSk7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyBhIGNvcHkgb2YgdGhlIHRlbnNvci4gU2VlIGB0Zi5jbG9uZWAgZm9yIGRldGFpbHMuXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgY2xvbmU8VCBleHRlbmRzIFRlbnNvcj4odGhpczogVCk6IFQge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIG9wSGFuZGxlci5jbG9uZSh0aGlzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXR1cm5zIGEgaHVtYW4tcmVhZGFibGUgZGVzY3JpcHRpb24gb2YgdGhlIHRlbnNvci4gVXNlZnVsIGZvciBsb2dnaW5nLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHRvU3RyaW5nKHZlcmJvc2UgPSBmYWxzZSk6IHN0cmluZyB7XG4gICAgY29uc3QgdmFscyA9IHRoaXMuZGF0YVN5bmMoKTtcbiAgICByZXR1cm4gdGVuc29yVG9TdHJpbmcodmFscywgdGhpcy5zaGFwZSwgdGhpcy5kdHlwZSwgdmVyYm9zZSk7XG4gIH1cblxuICBjYXN0PFQgZXh0ZW5kcyB0aGlzPihkdHlwZTogRGF0YVR5cGUpOiBUIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiBvcEhhbmRsZXIuY2FzdCh0aGlzIGFzIFQsIGR0eXBlKTtcbiAgfVxuICB2YXJpYWJsZSh0cmFpbmFibGUgPSB0cnVlLCBuYW1lPzogc3RyaW5nLCBkdHlwZT86IERhdGFUeXBlKTogVmFyaWFibGU8Uj4ge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIHRyYWNrZXJGbigpLm1ha2VWYXJpYWJsZSh0aGlzLCB0cmFpbmFibGUsIG5hbWUsIGR0eXBlKSBhc1xuICAgICAgICBWYXJpYWJsZTxSPjtcbiAgfVxufVxuXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoVGVuc29yLCBTeW1ib2wuaGFzSW5zdGFuY2UsIHtcbiAgdmFsdWU6IChpbnN0YW5jZTogVGVuc29yKSA9PiB7XG4gICAgLy8gSW1wbGVtZW50YXRpb24gbm90ZTogd2Ugc2hvdWxkIHVzZSBwcm9wZXJ0aWVzIG9mIHRoZSBvYmplY3QgdGhhdCB3aWxsIGJlXG4gICAgLy8gZGVmaW5lZCBiZWZvcmUgdGhlIGNvbnN0cnVjdG9yIGJvZHkgaGFzIGZpbmlzaGVkIGV4ZWN1dGluZyAobWV0aG9kcykuXG4gICAgLy8gVGhpcyBpcyBiZWNhdXNlIHdoZW4gdGhpcyBjb2RlIGlzIHRyYW5zcGlsZWQgYnkgYmFiZWwsIGJhYmVsIHdpbGwgY2FsbFxuICAgIC8vIGNsYXNzQ2FsbENoZWNrIGJlZm9yZSB0aGUgY29uc3RydWN0b3IgYm9keSBpcyBydW4uXG4gICAgLy8gU2VlIGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzMzODQgZm9yIGJhY2tzdG9yeS5cbiAgICByZXR1cm4gISFpbnN0YW5jZSAmJiBpbnN0YW5jZS5kYXRhICE9IG51bGwgJiYgaW5zdGFuY2UuZGF0YVN5bmMgIT0gbnVsbCAmJlxuICAgICAgICBpbnN0YW5jZS50aHJvd0lmRGlzcG9zZWQgIT0gbnVsbDtcbiAgfVxufSk7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRHbG9iYWxUZW5zb3JDbGFzcygpIHtcbiAgLy8gVXNlIGdldEdsb2JhbCBzbyB0aGF0IHdlIGNhbiBhdWdtZW50IHRoZSBUZW5zb3IgY2xhc3MgYWNyb3NzIHBhY2thZ2VcbiAgLy8gYm91bmRhcmllcyBiZWNhc2UgdGhlIG5vZGUgcmVzb2x1dGlvbiBhbGcgbWF5IHJlc3VsdCBpbiBkaWZmZXJlbnQgbW9kdWxlc1xuICAvLyBiZWluZyByZXR1cm5lZCBmb3IgdGhpcyBmaWxlIGRlcGVuZGluZyBvbiB0aGUgcGF0aCB0aGV5IGFyZSBsb2FkZWQgZnJvbS5cbiAgcmV0dXJuIGdldEdsb2JhbCgnVGVuc29yJywgKCkgPT4ge1xuICAgIHJldHVybiBUZW5zb3I7XG4gIH0pO1xufVxuXG4vLyBHbG9iYWwgc2lkZSBlZmZlY3QuIENhY2hlIGdsb2JhbCByZWZlcmVuY2UgdG8gVGVuc29yIGNsYXNzXG5nZXRHbG9iYWxUZW5zb3JDbGFzcygpO1xuXG5leHBvcnQgaW50ZXJmYWNlIE51bWVyaWNUZW5zb3I8UiBleHRlbmRzIFJhbmsgPSBSYW5rPiBleHRlbmRzIFRlbnNvcjxSPiB7XG4gIGR0eXBlOiBOdW1lcmljRGF0YVR5cGU7XG4gIGRhdGFTeW5jPEQgZXh0ZW5kcyBEYXRhVHlwZSA9IE51bWVyaWNEYXRhVHlwZT4oKTogRGF0YVR5cGVNYXBbRF07XG4gIGRhdGE8RCBleHRlbmRzIERhdGFUeXBlID0gTnVtZXJpY0RhdGFUeXBlPigpOiBQcm9taXNlPERhdGFUeXBlTWFwW0RdPjtcbiAgZGF0YVRvR1BVKG9wdGlvbnM/OiBEYXRhVG9HUFVPcHRpb25zKTogR1BVRGF0YTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBTdHJpbmdUZW5zb3I8UiBleHRlbmRzIFJhbmsgPSBSYW5rPiBleHRlbmRzIFRlbnNvcjxSPiB7XG4gIGR0eXBlOiAnc3RyaW5nJztcbiAgZGF0YVN5bmM8RCBleHRlbmRzIERhdGFUeXBlID0gJ3N0cmluZyc+KCk6IERhdGFUeXBlTWFwW0RdO1xuICBkYXRhPEQgZXh0ZW5kcyBEYXRhVHlwZSA9ICdzdHJpbmcnPigpOiBQcm9taXNlPERhdGFUeXBlTWFwW0RdPjtcbn1cblxuLyoqIEBkb2NsaW5rIFRlbnNvciAqL1xuZXhwb3J0IHR5cGUgU2NhbGFyID0gVGVuc29yPFJhbmsuUjA+O1xuLyoqIEBkb2NsaW5rIFRlbnNvciAqL1xuZXhwb3J0IHR5cGUgVGVuc29yMUQgPSBUZW5zb3I8UmFuay5SMT47XG4vKiogQGRvY2xpbmsgVGVuc29yICovXG5leHBvcnQgdHlwZSBUZW5zb3IyRCA9IFRlbnNvcjxSYW5rLlIyPjtcbi8qKiBAZG9jbGluayBUZW5zb3IgKi9cbmV4cG9ydCB0eXBlIFRlbnNvcjNEID0gVGVuc29yPFJhbmsuUjM+O1xuLyoqIEBkb2NsaW5rIFRlbnNvciAqL1xuZXhwb3J0IHR5cGUgVGVuc29yNEQgPSBUZW5zb3I8UmFuay5SND47XG4vKiogQGRvY2xpbmsgVGVuc29yICovXG5leHBvcnQgdHlwZSBUZW5zb3I1RCA9IFRlbnNvcjxSYW5rLlI1Pjtcbi8qKiBAZG9jbGluayBUZW5zb3IgKi9cbmV4cG9ydCB0eXBlIFRlbnNvcjZEID0gVGVuc29yPFJhbmsuUjY+O1xuXG4vKipcbiAqIEEgbXV0YWJsZSBgdGYuVGVuc29yYCwgdXNlZnVsIGZvciBwZXJzaXN0aW5nIHN0YXRlLCBlLmcuIGZvciB0cmFpbmluZy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAqL1xuZXhwb3J0IGNsYXNzIFZhcmlhYmxlPFIgZXh0ZW5kcyBSYW5rID0gUmFuaz4gZXh0ZW5kcyBUZW5zb3I8Uj4ge1xuICBuYW1lOiBzdHJpbmc7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBpbml0aWFsVmFsdWU6IFRlbnNvcjxSPiwgcHVibGljIHRyYWluYWJsZTogYm9vbGVhbiwgbmFtZTogc3RyaW5nLFxuICAgICAgdGVuc29ySWQ6IG51bWJlcikge1xuICAgIHN1cGVyKFxuICAgICAgICBpbml0aWFsVmFsdWUuc2hhcGUsIGluaXRpYWxWYWx1ZS5kdHlwZSwgaW5pdGlhbFZhbHVlLmRhdGFJZCwgdGVuc29ySWQpO1xuICAgIHRoaXMubmFtZSA9IG5hbWU7XG4gIH1cblxuICAvKipcbiAgICogQXNzaWduIGEgbmV3IGB0Zi5UZW5zb3JgIHRvIHRoaXMgdmFyaWFibGUuIFRoZSBuZXcgYHRmLlRlbnNvcmAgbXVzdCBoYXZlXG4gICAqIHRoZSBzYW1lIHNoYXBlIGFuZCBkdHlwZSBhcyB0aGUgb2xkIGB0Zi5UZW5zb3JgLlxuICAgKlxuICAgKiBAcGFyYW0gbmV3VmFsdWUgTmV3IHRlbnNvciB0byBiZSBhc3NpZ25lZCB0byB0aGlzIHZhcmlhYmxlLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFzc2lnbihuZXdWYWx1ZTogVGVuc29yPFI+KTogdm9pZCB7XG4gICAgaWYgKG5ld1ZhbHVlLmR0eXBlICE9PSB0aGlzLmR0eXBlKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYGR0eXBlIG9mIHRoZSBuZXcgdmFsdWUgKCR7bmV3VmFsdWUuZHR5cGV9KSBhbmQgYCArXG4gICAgICAgICAgYHByZXZpb3VzIHZhbHVlICgke3RoaXMuZHR5cGV9KSBtdXN0IG1hdGNoYCk7XG4gICAgfVxuICAgIGlmICghdXRpbC5hcnJheXNFcXVhbChuZXdWYWx1ZS5zaGFwZSwgdGhpcy5zaGFwZSkpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgc2hhcGUgb2YgdGhlIG5ldyB2YWx1ZSAoJHtuZXdWYWx1ZS5zaGFwZX0pIGFuZCBgICtcbiAgICAgICAgICBgcHJldmlvdXMgdmFsdWUgKCR7dGhpcy5zaGFwZX0pIG11c3QgbWF0Y2hgKTtcbiAgICB9XG4gICAgdHJhY2tlckZuKCkuZGlzcG9zZVRlbnNvcih0aGlzKTtcbiAgICB0aGlzLmRhdGFJZCA9IG5ld1ZhbHVlLmRhdGFJZDtcbiAgICB0cmFja2VyRm4oKS5pbmNSZWYodGhpcywgbnVsbCAvKiBiYWNrZW5kICovKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGRpc3Bvc2UoKTogdm9pZCB7XG4gICAgdHJhY2tlckZuKCkuZGlzcG9zZVZhcmlhYmxlKHRoaXMpO1xuICAgIHRoaXMuaXNEaXNwb3NlZEludGVybmFsID0gdHJ1ZTtcbiAgfVxufVxuXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoVmFyaWFibGUsIFN5bWJvbC5oYXNJbnN0YW5jZSwge1xuICB2YWx1ZTogKGluc3RhbmNlOiBWYXJpYWJsZSkgPT4ge1xuICAgIHJldHVybiBpbnN0YW5jZSBpbnN0YW5jZW9mIFRlbnNvciAmJiBpbnN0YW5jZS5hc3NpZ24gIT0gbnVsbCAmJlxuICAgICAgICBpbnN0YW5jZS5hc3NpZ24gaW5zdGFuY2VvZiBGdW5jdGlvbjtcbiAgfVxufSk7XG4iXX0=