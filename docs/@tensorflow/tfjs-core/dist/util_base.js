/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
/**
 * Shuffles the array in-place using Fisher-Yates algorithm.
 *
 * ```js
 * const a = [1, 2, 3, 4, 5];
 * tf.util.shuffle(a);
 * console.log(a);
 * ```
 *
 * @param array The array to shuffle in-place.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
// tslint:disable-next-line:no-any
export function shuffle(array) {
    let counter = array.length;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        index = (Math.random() * counter) | 0;
        // Decrease counter by 1
        counter--;
        // And swap the last element with it
        swap(array, counter, index);
    }
}
/**
 * Shuffles two arrays in-place the same way using Fisher-Yates algorithm.
 *
 * ```js
 * const a = [1,2,3,4,5];
 * const b = [11,22,33,44,55];
 * tf.util.shuffleCombo(a, b);
 * console.log(a, b);
 * ```
 *
 * @param array The first array to shuffle in-place.
 * @param array2 The second array to shuffle in-place with the same permutation
 *     as the first array.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function shuffleCombo(
// tslint:disable-next-line:no-any
array, 
// tslint:disable-next-line:no-any
array2) {
    if (array.length !== array2.length) {
        throw new Error(`Array sizes must match to be shuffled together ` +
            `First array length was ${array.length}` +
            `Second array length was ${array2.length}`);
    }
    let counter = array.length;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        index = (Math.random() * counter) | 0;
        // Decrease counter by 1
        counter--;
        // And swap the last element of each array with it
        swap(array, counter, index);
        swap(array2, counter, index);
    }
}
/** Clamps a value to a specified range. */
export function clamp(min, x, max) {
    return Math.max(min, Math.min(x, max));
}
export function nearestLargerEven(val) {
    return val % 2 === 0 ? val : val + 1;
}
export function swap(object, left, right) {
    const temp = object[left];
    object[left] = object[right];
    object[right] = temp;
}
export function sum(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum;
}
/**
 * Returns a sample from a uniform [a, b) distribution.
 *
 * @param a The minimum support (inclusive).
 * @param b The maximum support (exclusive).
 * @return A pseudorandom number on the half-open interval [a,b).
 */
export function randUniform(a, b) {
    const r = Math.random();
    return (b * r) + (1 - r) * a;
}
/** Returns the squared Euclidean distance between two vectors. */
export function distSquared(a, b) {
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = Number(a[i]) - Number(b[i]);
        result += diff * diff;
    }
    return result;
}
/**
 * Asserts that the expression is true. Otherwise throws an error with the
 * provided message.
 *
 * ```js
 * const x = 2;
 * tf.util.assert(x === 2, 'x is not 2');
 * ```
 *
 * @param expr The expression to assert (as a boolean).
 * @param msg A function that returns the message to report when throwing an
 *     error. We use a function for performance reasons.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function assert(expr, msg) {
    if (!expr) {
        throw new Error(typeof msg === 'string' ? msg : msg());
    }
}
export function assertShapesMatch(shapeA, shapeB, errorMessagePrefix = '') {
    assert(arraysEqual(shapeA, shapeB), () => errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
}
export function assertNonNull(a) {
    assert(a != null, () => `The input to the tensor constructor must be a non-null value.`);
}
/**
 * Returns the size (number of elements) of the tensor given its shape.
 *
 * ```js
 * const shape = [3, 4, 2];
 * const size = tf.util.sizeFromShape(shape);
 * console.log(size);
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function sizeFromShape(shape) {
    if (shape.length === 0) {
        // Scalar.
        return 1;
    }
    let size = shape[0];
    for (let i = 1; i < shape.length; i++) {
        size *= shape[i];
    }
    return size;
}
export function isScalarShape(shape) {
    return shape.length === 0;
}
export function arraysEqual(n1, n2) {
    if (n1 === n2) {
        return true;
    }
    if (n1 == null || n2 == null) {
        return false;
    }
    if (n1.length !== n2.length) {
        return false;
    }
    for (let i = 0; i < n1.length; i++) {
        if (n1[i] !== n2[i]) {
            return false;
        }
    }
    return true;
}
export function isInt(a) {
    return a % 1 === 0;
}
export function tanh(x) {
    // tslint:disable-next-line:no-any
    if (Math.tanh != null) {
        // tslint:disable-next-line:no-any
        return Math.tanh(x);
    }
    if (x === Infinity) {
        return 1;
    }
    else if (x === -Infinity) {
        return -1;
    }
    else {
        const e2x = Math.exp(2 * x);
        return (e2x - 1) / (e2x + 1);
    }
}
export function sizeToSquarishShape(size) {
    const width = Math.ceil(Math.sqrt(size));
    return [width, Math.ceil(size / width)];
}
/**
 * Creates a new array with randomized indices to a given quantity.
 *
 * ```js
 * const randomTen = tf.util.createShuffledIndices(10);
 * console.log(randomTen);
 * ```
 *
 * @param number Quantity of how many shuffled indices to create.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function createShuffledIndices(n) {
    const shuffledIndices = new Uint32Array(n);
    for (let i = 0; i < n; ++i) {
        shuffledIndices[i] = i;
    }
    shuffle(shuffledIndices);
    return shuffledIndices;
}
export function rightPad(a, size) {
    if (size <= a.length) {
        return a;
    }
    return a + ' '.repeat(size - a.length);
}
export function repeatedTry(checkFn, delayFn = (counter) => 0, maxCounter, scheduleFn) {
    return new Promise((resolve, reject) => {
        let tryCount = 0;
        const tryFn = () => {
            if (checkFn()) {
                resolve();
                return;
            }
            tryCount++;
            const nextBackoff = delayFn(tryCount);
            if (maxCounter != null && tryCount >= maxCounter) {
                reject();
                return;
            }
            if (scheduleFn != null) {
                scheduleFn(tryFn, nextBackoff);
            }
            else {
                // google3 does not allow assigning another variable to setTimeout.
                // Don't refactor this so scheduleFn has a default value of setTimeout.
                setTimeout(tryFn, nextBackoff);
            }
        };
        tryFn();
    });
}
/**
 * Given the full size of the array and a shape that may contain -1 as the
 * implicit dimension, returns the inferred shape where -1 is replaced.
 * E.g. For shape=[2, -1, 3] and size=24, it will return [2, 4, 3].
 *
 * @param shape The shape, which may contain -1 in some dimension.
 * @param size The full size (number of elements) of the array.
 * @return The inferred shape where -1 is replaced with the inferred size.
 */
export function inferFromImplicitShape(shape, size) {
    let shapeProd = 1;
    let implicitIdx = -1;
    for (let i = 0; i < shape.length; ++i) {
        if (shape[i] >= 0) {
            shapeProd *= shape[i];
        }
        else if (shape[i] === -1) {
            if (implicitIdx !== -1) {
                throw Error(`Shapes can only have 1 implicit size. ` +
                    `Found -1 at dim ${implicitIdx} and dim ${i}`);
            }
            implicitIdx = i;
        }
        else if (shape[i] < 0) {
            throw Error(`Shapes can not be < 0. Found ${shape[i]} at dim ${i}`);
        }
    }
    if (implicitIdx === -1) {
        if (size > 0 && size !== shapeProd) {
            throw Error(`Size(${size}) must match the product of shape ${shape}`);
        }
        return shape;
    }
    if (shapeProd === 0) {
        throw Error(`Cannot infer the missing size in [${shape}] when ` +
            `there are 0 elements`);
    }
    if (size % shapeProd !== 0) {
        throw Error(`The implicit shape can't be a fractional number. ` +
            `Got ${size} / ${shapeProd}`);
    }
    const newShape = shape.slice();
    newShape[implicitIdx] = size / shapeProd;
    return newShape;
}
export function parseAxisParam(axis, shape) {
    const rank = shape.length;
    // Normalize input
    axis = axis == null ? shape.map((s, i) => i) : [].concat(axis);
    // Check for valid range
    assert(axis.every(ax => ax >= -rank && ax < rank), () => `All values in axis param must be in range [-${rank}, ${rank}) but ` +
        `got axis ${axis}`);
    // Check for only integers
    assert(axis.every(ax => isInt(ax)), () => `All values in axis param must be integers but ` +
        `got axis ${axis}`);
    // Handle negative axis.
    return axis.map(a => a < 0 ? rank + a : a);
}
/** Reduces the shape by removing all dimensions of shape 1. */
export function squeezeShape(shape, axis) {
    const newShape = [];
    const keptDims = [];
    const isEmptyArray = axis != null && Array.isArray(axis) && axis.length === 0;
    const axes = (axis == null || isEmptyArray) ?
        null :
        parseAxisParam(axis, shape).sort();
    let j = 0;
    for (let i = 0; i < shape.length; ++i) {
        if (axes != null) {
            if (axes[j] === i && shape[i] !== 1) {
                throw new Error(`Can't squeeze axis ${i} since its dim '${shape[i]}' is not 1`);
            }
            if ((axes[j] == null || axes[j] > i) && shape[i] === 1) {
                newShape.push(shape[i]);
                keptDims.push(i);
            }
            if (axes[j] <= i) {
                j++;
            }
        }
        if (shape[i] !== 1) {
            newShape.push(shape[i]);
            keptDims.push(i);
        }
    }
    return { newShape, keptDims };
}
export function getTypedArrayFromDType(dtype, size) {
    let values = null;
    if (dtype == null || dtype === 'float32') {
        values = new Float32Array(size);
    }
    else if (dtype === 'int32') {
        values = new Int32Array(size);
    }
    else if (dtype === 'bool') {
        values = new Uint8Array(size);
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
    return values;
}
export function getArrayFromDType(dtype, size) {
    let values = null;
    if (dtype == null || dtype === 'float32') {
        values = new Float32Array(size);
    }
    else if (dtype === 'int32') {
        values = new Int32Array(size);
    }
    else if (dtype === 'bool') {
        values = new Uint8Array(size);
    }
    else if (dtype === 'string') {
        values = new Array(size);
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
    return values;
}
export function checkConversionForErrors(vals, dtype) {
    for (let i = 0; i < vals.length; i++) {
        const num = vals[i];
        if (isNaN(num) || !isFinite(num)) {
            throw Error(`A tensor of type ${dtype} being uploaded contains ${num}.`);
        }
    }
}
/** Returns true if the dtype is valid. */
export function isValidDtype(dtype) {
    return dtype === 'bool' || dtype === 'complex64' || dtype === 'float32' ||
        dtype === 'int32' || dtype === 'string';
}
/**
 * Returns true if the new type can't encode the old type without loss of
 * precision.
 */
export function hasEncodingLoss(oldType, newType) {
    if (newType === 'complex64') {
        return false;
    }
    if (newType === 'float32' && oldType !== 'complex64') {
        return false;
    }
    if (newType === 'int32' && oldType !== 'float32' && oldType !== 'complex64') {
        return false;
    }
    if (newType === 'bool' && oldType === 'bool') {
        return false;
    }
    return true;
}
export function bytesPerElement(dtype) {
    if (dtype === 'float32' || dtype === 'int32') {
        return 4;
    }
    else if (dtype === 'complex64') {
        return 8;
    }
    else if (dtype === 'bool') {
        return 1;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
/**
 * Returns the approximate number of bytes allocated in the string array - 2
 * bytes per character. Computing the exact bytes for a native string in JS
 * is not possible since it depends on the encoding of the html page that
 * serves the website.
 */
export function bytesFromStringArray(arr) {
    if (arr == null) {
        return 0;
    }
    let bytes = 0;
    arr.forEach(x => bytes += x.length);
    return bytes;
}
/** Returns true if the value is a string. */
export function isString(value) {
    return typeof value === 'string' || value instanceof String;
}
export function isBoolean(value) {
    return typeof value === 'boolean';
}
export function isNumber(value) {
    return typeof value === 'number';
}
export function inferDtype(values) {
    if (Array.isArray(values)) {
        return inferDtype(values[0]);
    }
    if (values instanceof Float32Array) {
        return 'float32';
    }
    else if (values instanceof Int32Array || values instanceof Uint8Array ||
        values instanceof Uint8ClampedArray) {
        return 'int32';
    }
    else if (isNumber(values)) {
        return 'float32';
    }
    else if (isString(values)) {
        return 'string';
    }
    else if (isBoolean(values)) {
        return 'bool';
    }
    return 'float32';
}
export function isFunction(f) {
    return !!(f && f.constructor && f.call && f.apply);
}
export function nearestDivisor(size, start) {
    for (let i = start; i < size; ++i) {
        if (size % i === 0) {
            return i;
        }
    }
    return size;
}
export function computeStrides(shape) {
    const rank = shape.length;
    if (rank < 2) {
        return [];
    }
    // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
    // strides.
    const strides = new Array(rank - 1);
    strides[rank - 2] = shape[rank - 1];
    for (let i = rank - 3; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}
function createNestedArray(offset, shape, a, isComplex = false) {
    const ret = new Array();
    if (shape.length === 1) {
        const d = shape[0] * (isComplex ? 2 : 1);
        for (let i = 0; i < d; i++) {
            ret[i] = a[offset + i];
        }
    }
    else {
        const d = shape[0];
        const rest = shape.slice(1);
        const len = rest.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
        for (let i = 0; i < d; i++) {
            ret[i] = createNestedArray(offset + i * len, rest, a, isComplex);
        }
    }
    return ret;
}
// Provide a nested array of TypedArray in given shape.
export function toNestedArray(shape, a, isComplex = false) {
    if (shape.length === 0) {
        // Scalar type should return a single number.
        return a[0];
    }
    const size = shape.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
    if (size === 0) {
        // A tensor with shape zero should be turned into empty list.
        return [];
    }
    if (size !== a.length) {
        throw new Error(`[${shape}] does not match the input size ${a.length}${isComplex ? ' for a complex tensor' : ''}.`);
    }
    return createNestedArray(0, shape, a, isComplex);
}
export function convertBackendValuesAndArrayBuffer(data, dtype) {
    // If is type Uint8Array[], return it directly.
    if (Array.isArray(data)) {
        return data;
    }
    if (dtype === 'float32') {
        return data instanceof Float32Array ? data : new Float32Array(data);
    }
    else if (dtype === 'int32') {
        return data instanceof Int32Array ? data : new Int32Array(data);
    }
    else if (dtype === 'bool' || dtype === 'string') {
        return Uint8Array.from(new Int32Array(data));
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
export function makeOnesTypedArray(size, dtype) {
    const array = makeZerosTypedArray(size, dtype);
    for (let i = 0; i < array.length; i++) {
        array[i] = 1;
    }
    return array;
}
export function makeZerosTypedArray(size, dtype) {
    if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
        return new Float32Array(size);
    }
    else if (dtype === 'int32') {
        return new Int32Array(size);
    }
    else if (dtype === 'bool') {
        return new Uint8Array(size);
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
/**
 * Make nested `TypedArray` filled with zeros.
 * @param shape The shape information for the nested array.
 * @param dtype dtype of the array element.
 */
export function makeZerosNestedTypedArray(shape, dtype) {
    const size = shape.reduce((prev, curr) => prev * curr, 1);
    if (dtype == null || dtype === 'float32') {
        return toNestedArray(shape, new Float32Array(size));
    }
    else if (dtype === 'int32') {
        return toNestedArray(shape, new Int32Array(size));
    }
    else if (dtype === 'bool') {
        return toNestedArray(shape, new Uint8Array(size));
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
export function assertNonNegativeIntegerDimensions(shape) {
    shape.forEach(dimSize => {
        assert(Number.isInteger(dimSize) && dimSize >= 0, () => `Tensor must have a shape comprised of positive integers but got ` +
            `shape [${shape}].`);
    });
}
/**
 * Computes flat index for a given location (multidimentionsal index) in a
 * Tensor/multidimensional array.
 *
 * @param locs Location in the tensor.
 * @param rank Rank of the tensor.
 * @param strides Tensor strides.
 */
export function locToIndex(locs, rank, strides) {
    if (rank === 0) {
        return 0;
    }
    else if (rank === 1) {
        return locs[0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
        index += strides[i] * locs[i];
    }
    return index;
}
/**
 * Computes the location (multidimensional index) in a
 * tensor/multidimentional array for a given flat index.
 *
 * @param index Index in flat array.
 * @param rank Rank of tensor.
 * @param strides Strides of tensor.
 */
export function indexToLoc(index, rank, strides) {
    if (rank === 0) {
        return [];
    }
    else if (rank === 1) {
        return [index];
    }
    const locs = new Array(rank);
    for (let i = 0; i < locs.length - 1; ++i) {
        locs[i] = Math.floor(index / strides[i]);
        index -= locs[i] * strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
}
/**
 * This method asserts whether an object is a Promise instance.
 * @param object
 */
// tslint:disable-next-line: no-any
export function isPromise(object) {
    //  We chose to not use 'obj instanceOf Promise' for two reasons:
    //  1. It only reliably works for es6 Promise, not other Promise
    //  implementations.
    //  2. It doesn't work with framework that uses zone.js. zone.js monkey
    //  patch the async calls, so it is possible the obj (patched) is
    //  comparing to a pre-patched Promise.
    return object && object.then && typeof object.then === 'function';
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidXRpbF9iYXNlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy91dGlsX2Jhc2UudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBSUg7Ozs7Ozs7Ozs7OztHQVlHO0FBQ0gsa0NBQWtDO0FBQ2xDLE1BQU0sVUFBVSxPQUFPLENBQUMsS0FDWTtJQUNsQyxJQUFJLE9BQU8sR0FBRyxLQUFLLENBQUMsTUFBTSxDQUFDO0lBQzNCLElBQUksS0FBSyxHQUFHLENBQUMsQ0FBQztJQUNkLHdDQUF3QztJQUN4QyxPQUFPLE9BQU8sR0FBRyxDQUFDLEVBQUU7UUFDbEIsc0JBQXNCO1FBQ3RCLEtBQUssR0FBRyxDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEMsd0JBQXdCO1FBQ3hCLE9BQU8sRUFBRSxDQUFDO1FBQ1Ysb0NBQW9DO1FBQ3BDLElBQUksQ0FBQyxLQUFLLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxDQUFDO0tBQzdCO0FBQ0gsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE1BQU0sVUFBVSxZQUFZO0FBQ3hCLGtDQUFrQztBQUNsQyxLQUFnRDtBQUNoRCxrQ0FBa0M7QUFDbEMsTUFBaUQ7SUFDbkQsSUFBSSxLQUFLLENBQUMsTUFBTSxLQUFLLE1BQU0sQ0FBQyxNQUFNLEVBQUU7UUFDbEMsTUFBTSxJQUFJLEtBQUssQ0FDWCxpREFBaUQ7WUFDakQsMEJBQTBCLEtBQUssQ0FBQyxNQUFNLEVBQUU7WUFDeEMsMkJBQTJCLE1BQU0sQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDO0tBQ2pEO0lBQ0QsSUFBSSxPQUFPLEdBQUcsS0FBSyxDQUFDLE1BQU0sQ0FBQztJQUMzQixJQUFJLEtBQUssR0FBRyxDQUFDLENBQUM7SUFDZCx3Q0FBd0M7SUFDeEMsT0FBTyxPQUFPLEdBQUcsQ0FBQyxF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QUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN6QyxLQUFLLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUMvQjtJQUNELElBQUksQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQyxHQUFHLEtBQUssQ0FBQztJQUM5QixPQUFPLElBQUksQ0FBQztBQUNkLENBQUM7QUFFRDs7O0dBR0c7QUFDSCxtQ0FBbUM7QUFDbkMsTUFBTSxVQUFVLFNBQVMsQ0FBQyxNQUFXO0lBQ25DLGlFQUFpRTtJQUNqRSxnRUFBZ0U7SUFDaEUsb0JBQW9CO0lBQ3BCLHVFQUF1RTtJQUN2RSxpRUFBaUU7SUFDakUsdUNBQXVDO0lBQ3ZDLE9BQU8sTUFBTSxJQUFJLE1BQU0sQ0FBQyxJQUFJLElBQUksT0FBTyxNQUFNLENBQUMsSUFBSSxLQUFLLFVBQVUsQ0FBQztBQUNwRSxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0JhY2tlbmRWYWx1ZXMsIERhdGFUeXBlLCBEYXRhVHlwZU1hcCwgRmxhdFZlY3RvciwgTnVtZXJpY0RhdGFUeXBlLCBUZW5zb3JMaWtlLCBUeXBlZEFycmF5LCBXZWJHTERhdGEsIFdlYkdQVURhdGF9IGZyb20gJy4vdHlwZXMnO1xuXG4vKipcbiAqIFNodWZmbGVzIHRoZSBhcnJheSBpbi1wbGFjZSB1c2luZyBGaXNoZXItWWF0ZXMgYWxnb3JpdGhtLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBhID0gWzEsIDIsIDMsIDQsIDVdO1xuICogdGYudXRpbC5zaHVmZmxlKGEpO1xuICogY29uc29sZS5sb2coYSk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gYXJyYXkgVGhlIGFycmF5IHRvIHNodWZmbGUgaW4tcGxhY2UuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1V0aWwnLCBuYW1lc3BhY2U6ICd1dGlsJ31cbiAqL1xuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuZXhwb3J0IGZ1bmN0aW9uIHNodWZmbGUoYXJyYXk6IGFueVtdfFVpbnQzMkFycmF5fEludDMyQXJyYXl8XG4gICAgICAgICAgICAgICAgICAgICAgICBGbG9hdDMyQXJyYXkpOiB2b2lkIHtcbiAgbGV0IGNvdW50ZXIgPSBhcnJheS5sZW5ndGg7XG4gIGxldCBpbmRleCA9IDA7XG4gIC8vIFdoaWxlIHRoZXJlIGFyZSBlbGVtZW50cyBpbiB0aGUgYXJyYXlcbiAgd2hpbGUgKGNvdW50ZXIgPiAwKSB7XG4gICAgLy8gUGljayBhIHJhbmRvbSBpbmRleFxuICAgIGluZGV4ID0gKE1hdGgucmFuZG9tKCkgKiBjb3VudGVyKSB8IDA7XG4gICAgLy8gRGVjcmVhc2UgY291bnRlciBieSAxXG4gICAgY291bnRlci0tO1xuICAgIC8vIEFuZCBzd2FwIHRoZSBsYXN0IGVsZW1lbnQgd2l0aCBpdFxuICAgIHN3YXAoYXJyYXksIGNvdW50ZXIsIGluZGV4KTtcbiAgfVxufVxuXG4vKipcbiAqIFNodWZmbGVzIHR3byBhcnJheXMgaW4tcGxhY2UgdGhlIHNhbWUgd2F5IHVzaW5nIEZpc2hlci1ZYXRlcyBhbGdvcml0aG0uXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSBbMSwyLDMsNCw1XTtcbiAqIGNvbnN0IGIgPSBbMTEsMjIsMzMsNDQsNTVdO1xuICogdGYudXRpbC5zaHVmZmxlQ29tYm8oYSwgYik7XG4gKiBjb25zb2xlLmxvZyhhLCBiKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBhcnJheSBUaGUgZmlyc3QgYXJyYXkgdG8gc2h1ZmZsZSBpbi1wbGFjZS5cbiAqIEBwYXJhbSBhcnJheTIgVGhlIHNlY29uZCBhcnJheSB0byBzaHVmZmxlIGluLXBsYWNlIHdpdGggdGhlIHNhbWUgcGVybXV0YXRpb25cbiAqICAgICBhcyB0aGUgZmlyc3QgYXJyYXkuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1V0aWwnLCBuYW1lc3BhY2U6ICd1dGlsJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHNodWZmbGVDb21ibyhcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgYXJyYXk6IGFueVtdfFVpbnQzMkFycmF5fEludDMyQXJyYXl8RmxvYXQzMkFycmF5LFxuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICBhcnJheTI6IGFueVtdfFVpbnQzMkFycmF5fEludDMyQXJyYXl8RmxvYXQzMkFycmF5KTogdm9pZCB7XG4gIGlmIChhcnJheS5sZW5ndGggIT09IGFycmF5Mi5sZW5ndGgpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBBcnJheSBzaXplcyBtdXN0IG1hdGNoIHRvIGJlIHNodWZmbGVkIHRvZ2V0aGVyIGAgK1xuICAgICAgICBgRmlyc3QgYXJyYXkgbGVuZ3RoIHdhcyAke2FycmF5Lmxlbmd0aH1gICtcbiAgICAgICAgYFNlY29uZCBhcnJheSBsZW5ndGggd2FzICR7YXJyYXkyLmxlbmd0aH1gKTtcbiAgfVxuICBsZXQgY291bnRlciA9IGFycmF5Lmxlbmd0aDtcbiAgbGV0IGluZGV4ID0gMDtcbiAgLy8gV2hpbGUgdGhlcmUgYXJlIGVsZW1lbnRzIGluIHRoZSBhcnJheVxuICB3aGlsZSAoY291bnRlciA+IDApIHtcbiAgICAvLyBQaWNrIGEgcmFuZG9tIGluZGV4XG4gICAgaW5kZXggPSAoTWF0aC5yYW5kb20oKSAqIGNvdW50ZXIpIHwgMDtcbiAgICAvLyBEZWNyZWFzZSBjb3VudGVyIGJ5IDFcbiAgICBjb3VudGVyLS07XG4gICAgLy8gQW5kIHN3YXAgdGhlIGxhc3QgZWxlbWVudCBvZiBlYWNoIGFycmF5IHdpdGggaXRcbiAgICBzd2FwKGFycmF5LCBjb3VudGVyLCBpbmRleCk7XG4gICAgc3dhcChhcnJheTIsIGNvdW50ZXIsIGluZGV4KTtcbiAgfVxufVxuXG4vKiogQ2xhbXBzIGEgdmFsdWUgdG8gYSBzcGVjaWZpZWQgcmFuZ2UuICovXG5leHBvcnQgZnVuY3Rpb24gY2xhbXAobWluOiBudW1iZXIsIHg6IG51bWJlciwgbWF4OiBudW1iZXIpOiBudW1iZXIge1xuICByZXR1cm4gTWF0aC5tYXgobWluLCBNYXRoLm1pbih4LCBtYXgpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG5lYXJlc3RMYXJnZXJFdmVuKHZhbDogbnVtYmVyKTogbnVtYmVyIHtcbiAgcmV0dXJuIHZhbCAlIDIgPT09IDAgPyB2YWwgOiB2YWwgKyAxO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gc3dhcDxUPihcbiAgICBvYmplY3Q6IHtbaW5kZXg6IG51bWJlcl06IFR9LCBsZWZ0OiBudW1iZXIsIHJpZ2h0OiBudW1iZXIpIHtcbiAgY29uc3QgdGVtcCA9IG9iamVjdFtsZWZ0XTtcbiAgb2JqZWN0W2xlZnRdID0gb2JqZWN0W3JpZ2h0XTtcbiAgb2JqZWN0W3JpZ2h0XSA9IHRlbXA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBzdW0oYXJyOiBudW1iZXJbXSk6IG51bWJlciB7XG4gIGxldCBzdW0gPSAwO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IGFyci5sZW5ndGg7IGkrKykge1xuICAgIHN1bSArPSBhcnJbaV07XG4gIH1cbiAgcmV0dXJuIHN1bTtcbn1cblxuLyoqXG4gKiBSZXR1cm5zIGEgc2FtcGxlIGZyb20gYSB1bmlmb3JtIFthLCBiKSBkaXN0cmlidXRpb24uXG4gKlxuICogQHBhcmFtIGEgVGhlIG1pbmltdW0gc3VwcG9ydCAoaW5jbHVzaXZlKS5cbiAqIEBwYXJhbSBiIFRoZSBtYXhpbXVtIHN1cHBvcnQgKGV4Y2x1c2l2ZSkuXG4gKiBAcmV0dXJuIEEgcHNldWRvcmFuZG9tIG51bWJlciBvbiB0aGUgaGFsZi1vcGVuIGludGVydmFsIFthLGIpLlxuICovXG5leHBvcnQgZnVuY3Rpb24gcmFuZFVuaWZvcm0oYTogbnVtYmVyLCBiOiBudW1iZXIpIHtcbiAgY29uc3QgciA9IE1hdGgucmFuZG9tKCk7XG4gIHJldHVybiAoYiAqIHIpICsgKDEgLSByKSAqIGE7XG59XG5cbi8qKiBSZXR1cm5zIHRoZSBzcXVhcmVkIEV1Y2xpZGVhbiBkaXN0YW5jZSBiZXR3ZWVuIHR3byB2ZWN0b3JzLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGRpc3RTcXVhcmVkKGE6IEZsYXRWZWN0b3IsIGI6IEZsYXRWZWN0b3IpOiBudW1iZXIge1xuICBsZXQgcmVzdWx0ID0gMDtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBhLmxlbmd0aDsgaSsrKSB7XG4gICAgY29uc3QgZGlmZiA9IE51bWJlcihhW2ldKSAtIE51bWJlcihiW2ldKTtcbiAgICByZXN1bHQgKz0gZGlmZiAqIGRpZmY7XG4gIH1cbiAgcmV0dXJuIHJlc3VsdDtcbn1cblxuLyoqXG4gKiBBc3NlcnRzIHRoYXQgdGhlIGV4cHJlc3Npb24gaXMgdHJ1ZS4gT3RoZXJ3aXNlIHRocm93cyBhbiBlcnJvciB3aXRoIHRoZVxuICogcHJvdmlkZWQgbWVzc2FnZS5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IDI7XG4gKiB0Zi51dGlsLmFzc2VydCh4ID09PSAyLCAneCBpcyBub3QgMicpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIGV4cHIgVGhlIGV4cHJlc3Npb24gdG8gYXNzZXJ0IChhcyBhIGJvb2xlYW4pLlxuICogQHBhcmFtIG1zZyBBIGZ1bmN0aW9uIHRoYXQgcmV0dXJucyB0aGUgbWVzc2FnZSB0byByZXBvcnQgd2hlbiB0aHJvd2luZyBhblxuICogICAgIGVycm9yLiBXZSB1c2UgYSBmdW5jdGlvbiBmb3IgcGVyZm9ybWFuY2UgcmVhc29ucy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVXRpbCcsIG5hbWVzcGFjZTogJ3V0aWwnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0KGV4cHI6IGJvb2xlYW4sIG1zZzogKCkgPT4gc3RyaW5nKSB7XG4gIGlmICghZXhwcikge1xuICAgIHRocm93IG5ldyBFcnJvcih0eXBlb2YgbXNnID09PSAnc3RyaW5nJyA/IG1zZyA6IG1zZygpKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0U2hhcGVzTWF0Y2goXG4gICAgc2hhcGVBOiBudW1iZXJbXSwgc2hhcGVCOiBudW1iZXJbXSwgZXJyb3JNZXNzYWdlUHJlZml4ID0gJycpOiB2b2lkIHtcbiAgYXNzZXJ0KFxuICAgICAgYXJyYXlzRXF1YWwoc2hhcGVBLCBzaGFwZUIpLFxuICAgICAgKCkgPT4gZXJyb3JNZXNzYWdlUHJlZml4ICsgYCBTaGFwZXMgJHtzaGFwZUF9IGFuZCAke3NoYXBlQn0gbXVzdCBtYXRjaGApO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0Tm9uTnVsbChhOiBUZW5zb3JMaWtlKTogdm9pZCB7XG4gIGFzc2VydChcbiAgICAgIGEgIT0gbnVsbCxcbiAgICAgICgpID0+IGBUaGUgaW5wdXQgdG8gdGhlIHRlbnNvciBjb25zdHJ1Y3RvciBtdXN0IGJlIGEgbm9uLW51bGwgdmFsdWUuYCk7XG59XG5cbi8qKlxuICogUmV0dXJucyB0aGUgc2l6ZSAobnVtYmVyIG9mIGVsZW1lbnRzKSBvZiB0aGUgdGVuc29yIGdpdmVuIGl0cyBzaGFwZS5cbiAqXG4gKiBgYGBqc1xuICogY29uc3Qgc2hhcGUgPSBbMywgNCwgMl07XG4gKiBjb25zdCBzaXplID0gdGYudXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcbiAqIGNvbnNvbGUubG9nKHNpemUpO1xuICogYGBgXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1V0aWwnLCBuYW1lc3BhY2U6ICd1dGlsJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHNpemVGcm9tU2hhcGUoc2hhcGU6IG51bWJlcltdKTogbnVtYmVyIHtcbiAgaWYgKHNoYXBlLmxlbmd0aCA9PT0gMCkge1xuICAgIC8vIFNjYWxhci5cbiAgICByZXR1cm4gMTtcbiAgfVxuICBsZXQgc2l6ZSA9IHNoYXBlWzBdO1xuICBmb3IgKGxldCBpID0gMTsgaSA8IHNoYXBlLmxlbmd0aDsgaSsrKSB7XG4gICAgc2l6ZSAqPSBzaGFwZVtpXTtcbiAgfVxuICByZXR1cm4gc2l6ZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGlzU2NhbGFyU2hhcGUoc2hhcGU6IG51bWJlcltdKTogYm9vbGVhbiB7XG4gIHJldHVybiBzaGFwZS5sZW5ndGggPT09IDA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBhcnJheXNFcXVhbChuMTogRmxhdFZlY3RvciwgbjI6IEZsYXRWZWN0b3IpIHtcbiAgaWYgKG4xID09PSBuMikge1xuICAgIHJldHVybiB0cnVlO1xuICB9XG4gIGlmIChuMSA9PSBudWxsIHx8IG4yID09IG51bGwpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cblxuICBpZiAobjEubGVuZ3RoICE9PSBuMi5sZW5ndGgpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuMS5sZW5ndGg7IGkrKykge1xuICAgIGlmIChuMVtpXSAhPT0gbjJbaV0pIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHRydWU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc0ludChhOiBudW1iZXIpOiBib29sZWFuIHtcbiAgcmV0dXJuIGEgJSAxID09PSAwO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdGFuaCh4OiBudW1iZXIpOiBudW1iZXIge1xuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gIGlmICgoTWF0aCBhcyBhbnkpLnRhbmggIT0gbnVsbCkge1xuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICByZXR1cm4gKE1hdGggYXMgYW55KS50YW5oKHgpO1xuICB9XG4gIGlmICh4ID09PSBJbmZpbml0eSkge1xuICAgIHJldHVybiAxO1xuICB9IGVsc2UgaWYgKHggPT09IC1JbmZpbml0eSkge1xuICAgIHJldHVybiAtMTtcbiAgfSBlbHNlIHtcbiAgICBjb25zdCBlMnggPSBNYXRoLmV4cCgyICogeCk7XG4gICAgcmV0dXJuIChlMnggLSAxKSAvIChlMnggKyAxKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gc2l6ZVRvU3F1YXJpc2hTaGFwZShzaXplOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgY29uc3Qgd2lkdGggPSBNYXRoLmNlaWwoTWF0aC5zcXJ0KHNpemUpKTtcbiAgcmV0dXJuIFt3aWR0aCwgTWF0aC5jZWlsKHNpemUgLyB3aWR0aCldO1xufVxuXG4vKipcbiAqIENyZWF0ZXMgYSBuZXcgYXJyYXkgd2l0aCByYW5kb21pemVkIGluZGljZXMgdG8gYSBnaXZlbiBxdWFudGl0eS5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgcmFuZG9tVGVuID0gdGYudXRpbC5jcmVhdGVTaHVmZmxlZEluZGljZXMoMTApO1xuICogY29uc29sZS5sb2cocmFuZG9tVGVuKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBudW1iZXIgUXVhbnRpdHkgb2YgaG93IG1hbnkgc2h1ZmZsZWQgaW5kaWNlcyB0byBjcmVhdGUuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1V0aWwnLCBuYW1lc3BhY2U6ICd1dGlsJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVNodWZmbGVkSW5kaWNlcyhuOiBudW1iZXIpOiBVaW50MzJBcnJheSB7XG4gIGNvbnN0IHNodWZmbGVkSW5kaWNlcyA9IG5ldyBVaW50MzJBcnJheShuKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuOyArK2kpIHtcbiAgICBzaHVmZmxlZEluZGljZXNbaV0gPSBpO1xuICB9XG4gIHNodWZmbGUoc2h1ZmZsZWRJbmRpY2VzKTtcbiAgcmV0dXJuIHNodWZmbGVkSW5kaWNlcztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHJpZ2h0UGFkKGE6IHN0cmluZywgc2l6ZTogbnVtYmVyKTogc3RyaW5nIHtcbiAgaWYgKHNpemUgPD0gYS5sZW5ndGgpIHtcbiAgICByZXR1cm4gYTtcbiAgfVxuICByZXR1cm4gYSArICcgJy5yZXBlYXQoc2l6ZSAtIGEubGVuZ3RoKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHJlcGVhdGVkVHJ5KFxuICAgIGNoZWNrRm46ICgpID0+IGJvb2xlYW4sIGRlbGF5Rm4gPSAoY291bnRlcjogbnVtYmVyKSA9PiAwLFxuICAgIG1heENvdW50ZXI/OiBudW1iZXIsXG4gICAgc2NoZWR1bGVGbj86IChmdW5jdGlvblJlZjogRnVuY3Rpb24sIGRlbGF5OiBudW1iZXIpID0+XG4gICAgICAgIHZvaWQpOiBQcm9taXNlPHZvaWQ+IHtcbiAgcmV0dXJuIG5ldyBQcm9taXNlPHZvaWQ+KChyZXNvbHZlLCByZWplY3QpID0+IHtcbiAgICBsZXQgdHJ5Q291bnQgPSAwO1xuXG4gICAgY29uc3QgdHJ5Rm4gPSAoKSA9PiB7XG4gICAgICBpZiAoY2hlY2tGbigpKSB7XG4gICAgICAgIHJlc29sdmUoKTtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuXG4gICAgICB0cnlDb3VudCsrO1xuXG4gICAgICBjb25zdCBuZXh0QmFja29mZiA9IGRlbGF5Rm4odHJ5Q291bnQpO1xuXG4gICAgICBpZiAobWF4Q291bnRlciAhPSBudWxsICYmIHRyeUNvdW50ID49IG1heENvdW50ZXIpIHtcbiAgICAgICAgcmVqZWN0KCk7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgaWYgKHNjaGVkdWxlRm4gIT0gbnVsbCkge1xuICAgICAgICBzY2hlZHVsZUZuKHRyeUZuLCBuZXh0QmFja29mZik7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICAvLyBnb29nbGUzIGRvZXMgbm90IGFsbG93IGFzc2lnbmluZyBhbm90aGVyIHZhcmlhYmxlIHRvIHNldFRpbWVvdXQuXG4gICAgICAgIC8vIERvbid0IHJlZmFjdG9yIHRoaXMgc28gc2NoZWR1bGVGbiBoYXMgYSBkZWZhdWx0IHZhbHVlIG9mIHNldFRpbWVvdXQuXG4gICAgICAgIHNldFRpbWVvdXQodHJ5Rm4sIG5leHRCYWNrb2ZmKTtcbiAgICAgIH1cbiAgICB9O1xuXG4gICAgdHJ5Rm4oKTtcbiAgfSk7XG59XG5cbi8qKlxuICogR2l2ZW4gdGhlIGZ1bGwgc2l6ZSBvZiB0aGUgYXJyYXkgYW5kIGEgc2hhcGUgdGhhdCBtYXkgY29udGFpbiAtMSBhcyB0aGVcbiAqIGltcGxpY2l0IGRpbWVuc2lvbiwgcmV0dXJucyB0aGUgaW5mZXJyZWQgc2hhcGUgd2hlcmUgLTEgaXMgcmVwbGFjZWQuXG4gKiBFLmcuIEZvciBzaGFwZT1bMiwgLTEsIDNdIGFuZCBzaXplPTI0LCBpdCB3aWxsIHJldHVybiBbMiwgNCwgM10uXG4gKlxuICogQHBhcmFtIHNoYXBlIFRoZSBzaGFwZSwgd2hpY2ggbWF5IGNvbnRhaW4gLTEgaW4gc29tZSBkaW1lbnNpb24uXG4gKiBAcGFyYW0gc2l6ZSBUaGUgZnVsbCBzaXplIChudW1iZXIgb2YgZWxlbWVudHMpIG9mIHRoZSBhcnJheS5cbiAqIEByZXR1cm4gVGhlIGluZmVycmVkIHNoYXBlIHdoZXJlIC0xIGlzIHJlcGxhY2VkIHdpdGggdGhlIGluZmVycmVkIHNpemUuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpbmZlckZyb21JbXBsaWNpdFNoYXBlKFxuICAgIHNoYXBlOiBudW1iZXJbXSwgc2l6ZTogbnVtYmVyKTogbnVtYmVyW10ge1xuICBsZXQgc2hhcGVQcm9kID0gMTtcbiAgbGV0IGltcGxpY2l0SWR4ID0gLTE7XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBzaGFwZS5sZW5ndGg7ICsraSkge1xuICAgIGlmIChzaGFwZVtpXSA+PSAwKSB7XG4gICAgICBzaGFwZVByb2QgKj0gc2hhcGVbaV07XG4gICAgfSBlbHNlIGlmIChzaGFwZVtpXSA9PT0gLTEpIHtcbiAgICAgIGlmIChpbXBsaWNpdElkeCAhPT0gLTEpIHtcbiAgICAgICAgdGhyb3cgRXJyb3IoXG4gICAgICAgICAgICBgU2hhcGVzIGNhbiBvbmx5IGhhdmUgMSBpbXBsaWNpdCBzaXplLiBgICtcbiAgICAgICAgICAgIGBGb3VuZCAtMSBhdCBkaW0gJHtpbXBsaWNpdElkeH0gYW5kIGRpbSAke2l9YCk7XG4gICAgICB9XG4gICAgICBpbXBsaWNpdElkeCA9IGk7XG4gICAgfSBlbHNlIGlmIChzaGFwZVtpXSA8IDApIHtcbiAgICAgIHRocm93IEVycm9yKGBTaGFwZXMgY2FuIG5vdCBiZSA8IDAuIEZvdW5kICR7c2hhcGVbaV19IGF0IGRpbSAke2l9YCk7XG4gICAgfVxuICB9XG5cbiAgaWYgKGltcGxpY2l0SWR4ID09PSAtMSkge1xuICAgIGlmIChzaXplID4gMCAmJiBzaXplICE9PSBzaGFwZVByb2QpIHtcbiAgICAgIHRocm93IEVycm9yKGBTaXplKCR7c2l6ZX0pIG11c3QgbWF0Y2ggdGhlIHByb2R1Y3Qgb2Ygc2hhcGUgJHtzaGFwZX1gKTtcbiAgICB9XG4gICAgcmV0dXJuIHNoYXBlO1xuICB9XG5cbiAgaWYgKHNoYXBlUHJvZCA9PT0gMCkge1xuICAgIHRocm93IEVycm9yKFxuICAgICAgICBgQ2Fubm90IGluZmVyIHRoZSBtaXNzaW5nIHNpemUgaW4gWyR7c2hhcGV9XSB3aGVuIGAgK1xuICAgICAgICBgdGhlcmUgYXJlIDAgZWxlbWVudHNgKTtcbiAgfVxuICBpZiAoc2l6ZSAlIHNoYXBlUHJvZCAhPT0gMCkge1xuICAgIHRocm93IEVycm9yKFxuICAgICAgICBgVGhlIGltcGxpY2l0IHNoYXBlIGNhbid0IGJlIGEgZnJhY3Rpb25hbCBudW1iZXIuIGAgK1xuICAgICAgICBgR290ICR7c2l6ZX0gLyAke3NoYXBlUHJvZH1gKTtcbiAgfVxuXG4gIGNvbnN0IG5ld1NoYXBlID0gc2hhcGUuc2xpY2UoKTtcbiAgbmV3U2hhcGVbaW1wbGljaXRJZHhdID0gc2l6ZSAvIHNoYXBlUHJvZDtcbiAgcmV0dXJuIG5ld1NoYXBlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gcGFyc2VBeGlzUGFyYW0oXG4gICAgYXhpczogbnVtYmVyfG51bWJlcltdLCBzaGFwZTogbnVtYmVyW10pOiBudW1iZXJbXSB7XG4gIGNvbnN0IHJhbmsgPSBzaGFwZS5sZW5ndGg7XG5cbiAgLy8gTm9ybWFsaXplIGlucHV0XG4gIGF4aXMgPSBheGlzID09IG51bGwgPyBzaGFwZS5tYXAoKHMsIGkpID0+IGkpIDogW10uY29uY2F0KGF4aXMpO1xuXG4gIC8vIENoZWNrIGZvciB2YWxpZCByYW5nZVxuICBhc3NlcnQoXG4gICAgICBheGlzLmV2ZXJ5KGF4ID0+IGF4ID49IC1yYW5rICYmIGF4IDwgcmFuayksXG4gICAgICAoKSA9PlxuICAgICAgICAgIGBBbGwgdmFsdWVzIGluIGF4aXMgcGFyYW0gbXVzdCBiZSBpbiByYW5nZSBbLSR7cmFua30sICR7cmFua30pIGJ1dCBgICtcbiAgICAgICAgICBgZ290IGF4aXMgJHtheGlzfWApO1xuXG4gIC8vIENoZWNrIGZvciBvbmx5IGludGVnZXJzXG4gIGFzc2VydChcbiAgICAgIGF4aXMuZXZlcnkoYXggPT4gaXNJbnQoYXgpKSxcbiAgICAgICgpID0+IGBBbGwgdmFsdWVzIGluIGF4aXMgcGFyYW0gbXVzdCBiZSBpbnRlZ2VycyBidXQgYCArXG4gICAgICAgICAgYGdvdCBheGlzICR7YXhpc31gKTtcblxuICAvLyBIYW5kbGUgbmVnYXRpdmUgYXhpcy5cbiAgcmV0dXJuIGF4aXMubWFwKGEgPT4gYSA8IDAgPyByYW5rICsgYSA6IGEpO1xufVxuXG4vKiogUmVkdWNlcyB0aGUgc2hhcGUgYnkgcmVtb3ZpbmcgYWxsIGRpbWVuc2lvbnMgb2Ygc2hhcGUgMS4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzcXVlZXplU2hhcGUoc2hhcGU6IG51bWJlcltdLCBheGlzPzogbnVtYmVyW10pOlxuICAgIHtuZXdTaGFwZTogbnVtYmVyW10sIGtlcHREaW1zOiBudW1iZXJbXX0ge1xuICBjb25zdCBuZXdTaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgY29uc3Qga2VwdERpbXM6IG51bWJlcltdID0gW107XG4gIGNvbnN0IGlzRW1wdHlBcnJheSA9IGF4aXMgIT0gbnVsbCAmJiBBcnJheS5pc0FycmF5KGF4aXMpICYmIGF4aXMubGVuZ3RoID09PSAwO1xuICBjb25zdCBheGVzID0gKGF4aXMgPT0gbnVsbCB8fCBpc0VtcHR5QXJyYXkpID9cbiAgICAgIG51bGwgOlxuICAgICAgcGFyc2VBeGlzUGFyYW0oYXhpcywgc2hhcGUpLnNvcnQoKTtcbiAgbGV0IGogPSAwO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IHNoYXBlLmxlbmd0aDsgKytpKSB7XG4gICAgaWYgKGF4ZXMgIT0gbnVsbCkge1xuICAgICAgaWYgKGF4ZXNbal0gPT09IGkgJiYgc2hhcGVbaV0gIT09IDEpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgYENhbid0IHNxdWVlemUgYXhpcyAke2l9IHNpbmNlIGl0cyBkaW0gJyR7c2hhcGVbaV19JyBpcyBub3QgMWApO1xuICAgICAgfVxuICAgICAgaWYgKChheGVzW2pdID09IG51bGwgfHwgYXhlc1tqXSA+IGkpICYmIHNoYXBlW2ldID09PSAxKSB7XG4gICAgICAgIG5ld1NoYXBlLnB1c2goc2hhcGVbaV0pO1xuICAgICAgICBrZXB0RGltcy5wdXNoKGkpO1xuICAgICAgfVxuICAgICAgaWYgKGF4ZXNbal0gPD0gaSkge1xuICAgICAgICBqKys7XG4gICAgICB9XG4gICAgfVxuICAgIGlmIChzaGFwZVtpXSAhPT0gMSkge1xuICAgICAgbmV3U2hhcGUucHVzaChzaGFwZVtpXSk7XG4gICAgICBrZXB0RGltcy5wdXNoKGkpO1xuICAgIH1cbiAgfVxuICByZXR1cm4ge25ld1NoYXBlLCBrZXB0RGltc307XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRUeXBlZEFycmF5RnJvbURUeXBlPEQgZXh0ZW5kcyBOdW1lcmljRGF0YVR5cGU+KFxuICAgIGR0eXBlOiBELCBzaXplOiBudW1iZXIpOiBEYXRhVHlwZU1hcFtEXSB7XG4gIGxldCB2YWx1ZXMgPSBudWxsO1xuICBpZiAoZHR5cGUgPT0gbnVsbCB8fCBkdHlwZSA9PT0gJ2Zsb2F0MzInKSB7XG4gICAgdmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShzaXplKTtcbiAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ2ludDMyJykge1xuICAgIHZhbHVlcyA9IG5ldyBJbnQzMkFycmF5KHNpemUpO1xuICB9IGVsc2UgaWYgKGR0eXBlID09PSAnYm9vbCcpIHtcbiAgICB2YWx1ZXMgPSBuZXcgVWludDhBcnJheShzaXplKTtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVua25vd24gZGF0YSB0eXBlICR7ZHR5cGV9YCk7XG4gIH1cbiAgcmV0dXJuIHZhbHVlcyBhcyBEYXRhVHlwZU1hcFtEXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldEFycmF5RnJvbURUeXBlPEQgZXh0ZW5kcyBEYXRhVHlwZT4oXG4gICAgZHR5cGU6IEQsIHNpemU6IG51bWJlcik6IERhdGFUeXBlTWFwW0RdIHtcbiAgbGV0IHZhbHVlcyA9IG51bGw7XG4gIGlmIChkdHlwZSA9PSBudWxsIHx8IGR0eXBlID09PSAnZmxvYXQzMicpIHtcbiAgICB2YWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHNpemUpO1xuICB9IGVsc2UgaWYgKGR0eXBlID09PSAnaW50MzInKSB7XG4gICAgdmFsdWVzID0gbmV3IEludDMyQXJyYXkoc2l6ZSk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdib29sJykge1xuICAgIHZhbHVlcyA9IG5ldyBVaW50OEFycmF5KHNpemUpO1xuICB9IGVsc2UgaWYgKGR0eXBlID09PSAnc3RyaW5nJykge1xuICAgIHZhbHVlcyA9IG5ldyBBcnJheTwnc3RyaW5nJz4oc2l6ZSk7XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBVbmtub3duIGRhdGEgdHlwZSAke2R0eXBlfWApO1xuICB9XG4gIHJldHVybiB2YWx1ZXMgYXMgRGF0YVR5cGVNYXBbRF07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjaGVja0NvbnZlcnNpb25Gb3JFcnJvcnM8RCBleHRlbmRzIERhdGFUeXBlPihcbiAgICB2YWxzOiBEYXRhVHlwZU1hcFtEXXxudW1iZXJbXSwgZHR5cGU6IEQpOiB2b2lkIHtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWxzLmxlbmd0aDsgaSsrKSB7XG4gICAgY29uc3QgbnVtID0gdmFsc1tpXSBhcyBudW1iZXI7XG4gICAgaWYgKGlzTmFOKG51bSkgfHwgIWlzRmluaXRlKG51bSkpIHtcbiAgICAgIHRocm93IEVycm9yKGBBIHRlbnNvciBvZiB0eXBlICR7ZHR5cGV9IGJlaW5nIHVwbG9hZGVkIGNvbnRhaW5zICR7bnVtfS5gKTtcbiAgICB9XG4gIH1cbn1cblxuLyoqIFJldHVybnMgdHJ1ZSBpZiB0aGUgZHR5cGUgaXMgdmFsaWQuICovXG5leHBvcnQgZnVuY3Rpb24gaXNWYWxpZER0eXBlKGR0eXBlOiBEYXRhVHlwZSk6IGJvb2xlYW4ge1xuICByZXR1cm4gZHR5cGUgPT09ICdib29sJyB8fCBkdHlwZSA9PT0gJ2NvbXBsZXg2NCcgfHwgZHR5cGUgPT09ICdmbG9hdDMyJyB8fFxuICAgICAgZHR5cGUgPT09ICdpbnQzMicgfHwgZHR5cGUgPT09ICdzdHJpbmcnO1xufVxuXG4vKipcbiAqIFJldHVybnMgdHJ1ZSBpZiB0aGUgbmV3IHR5cGUgY2FuJ3QgZW5jb2RlIHRoZSBvbGQgdHlwZSB3aXRob3V0IGxvc3Mgb2ZcbiAqIHByZWNpc2lvbi5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGhhc0VuY29kaW5nTG9zcyhvbGRUeXBlOiBEYXRhVHlwZSwgbmV3VHlwZTogRGF0YVR5cGUpOiBib29sZWFuIHtcbiAgaWYgKG5ld1R5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGlmIChuZXdUeXBlID09PSAnZmxvYXQzMicgJiYgb2xkVHlwZSAhPT0gJ2NvbXBsZXg2NCcpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgaWYgKG5ld1R5cGUgPT09ICdpbnQzMicgJiYgb2xkVHlwZSAhPT0gJ2Zsb2F0MzInICYmIG9sZFR5cGUgIT09ICdjb21wbGV4NjQnKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGlmIChuZXdUeXBlID09PSAnYm9vbCcgJiYgb2xkVHlwZSA9PT0gJ2Jvb2wnKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIHJldHVybiB0cnVlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYnl0ZXNQZXJFbGVtZW50KGR0eXBlOiBEYXRhVHlwZSk6IG51bWJlciB7XG4gIGlmIChkdHlwZSA9PT0gJ2Zsb2F0MzInIHx8IGR0eXBlID09PSAnaW50MzInKSB7XG4gICAgcmV0dXJuIDQ7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgcmV0dXJuIDg7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdib29sJykge1xuICAgIHJldHVybiAxO1xuICB9IGVsc2Uge1xuICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biBkdHlwZSAke2R0eXBlfWApO1xuICB9XG59XG5cbi8qKlxuICogUmV0dXJucyB0aGUgYXBwcm94aW1hdGUgbnVtYmVyIG9mIGJ5dGVzIGFsbG9jYXRlZCBpbiB0aGUgc3RyaW5nIGFycmF5IC0gMlxuICogYnl0ZXMgcGVyIGNoYXJhY3Rlci4gQ29tcHV0aW5nIHRoZSBleGFjdCBieXRlcyBmb3IgYSBuYXRpdmUgc3RyaW5nIGluIEpTXG4gKiBpcyBub3QgcG9zc2libGUgc2luY2UgaXQgZGVwZW5kcyBvbiB0aGUgZW5jb2Rpbmcgb2YgdGhlIGh0bWwgcGFnZSB0aGF0XG4gKiBzZXJ2ZXMgdGhlIHdlYnNpdGUuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBieXRlc0Zyb21TdHJpbmdBcnJheShhcnI6IFVpbnQ4QXJyYXlbXSk6IG51bWJlciB7XG4gIGlmIChhcnIgPT0gbnVsbCkge1xuICAgIHJldHVybiAwO1xuICB9XG4gIGxldCBieXRlcyA9IDA7XG4gIGFyci5mb3JFYWNoKHggPT4gYnl0ZXMgKz0geC5sZW5ndGgpO1xuICByZXR1cm4gYnl0ZXM7XG59XG5cbi8qKiBSZXR1cm5zIHRydWUgaWYgdGhlIHZhbHVlIGlzIGEgc3RyaW5nLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGlzU3RyaW5nKHZhbHVlOiB7fSk6IHZhbHVlIGlzIHN0cmluZyB7XG4gIHJldHVybiB0eXBlb2YgdmFsdWUgPT09ICdzdHJpbmcnIHx8IHZhbHVlIGluc3RhbmNlb2YgU3RyaW5nO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNCb29sZWFuKHZhbHVlOiB7fSk6IGJvb2xlYW4ge1xuICByZXR1cm4gdHlwZW9mIHZhbHVlID09PSAnYm9vbGVhbic7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc051bWJlcih2YWx1ZToge30pOiBib29sZWFuIHtcbiAgcmV0dXJuIHR5cGVvZiB2YWx1ZSA9PT0gJ251bWJlcic7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpbmZlckR0eXBlKHZhbHVlczogVGVuc29yTGlrZXxXZWJHTERhdGF8V2ViR1BVRGF0YSk6IERhdGFUeXBlIHtcbiAgaWYgKEFycmF5LmlzQXJyYXkodmFsdWVzKSkge1xuICAgIHJldHVybiBpbmZlckR0eXBlKHZhbHVlc1swXSk7XG4gIH1cbiAgaWYgKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkge1xuICAgIHJldHVybiAnZmxvYXQzMic7XG4gIH0gZWxzZSBpZiAoXG4gICAgICB2YWx1ZXMgaW5zdGFuY2VvZiBJbnQzMkFycmF5IHx8IHZhbHVlcyBpbnN0YW5jZW9mIFVpbnQ4QXJyYXkgfHxcbiAgICAgIHZhbHVlcyBpbnN0YW5jZW9mIFVpbnQ4Q2xhbXBlZEFycmF5KSB7XG4gICAgcmV0dXJuICdpbnQzMic7XG4gIH0gZWxzZSBpZiAoaXNOdW1iZXIodmFsdWVzKSkge1xuICAgIHJldHVybiAnZmxvYXQzMic7XG4gIH0gZWxzZSBpZiAoaXNTdHJpbmcodmFsdWVzKSkge1xuICAgIHJldHVybiAnc3RyaW5nJztcbiAgfSBlbHNlIGlmIChpc0Jvb2xlYW4odmFsdWVzKSkge1xuICAgIHJldHVybiAnYm9vbCc7XG4gIH1cbiAgcmV0dXJuICdmbG9hdDMyJztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGlzRnVuY3Rpb24oZjogRnVuY3Rpb24pIHtcbiAgcmV0dXJuICEhKGYgJiYgZi5jb25zdHJ1Y3RvciAmJiBmLmNhbGwgJiYgZi5hcHBseSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBuZWFyZXN0RGl2aXNvcihzaXplOiBudW1iZXIsIHN0YXJ0OiBudW1iZXIpOiBudW1iZXIge1xuICBmb3IgKGxldCBpID0gc3RhcnQ7IGkgPCBzaXplOyArK2kpIHtcbiAgICBpZiAoc2l6ZSAlIGkgPT09IDApIHtcbiAgICAgIHJldHVybiBpO1xuICAgIH1cbiAgfVxuICByZXR1cm4gc2l6ZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVTdHJpZGVzKHNoYXBlOiBudW1iZXJbXSk6IG51bWJlcltdIHtcbiAgY29uc3QgcmFuayA9IHNoYXBlLmxlbmd0aDtcbiAgaWYgKHJhbmsgPCAyKSB7XG4gICAgcmV0dXJuIFtdO1xuICB9XG5cbiAgLy8gTGFzdCBkaW1lbnNpb24gaGFzIGltcGxpY2l0IHN0cmlkZSBvZiAxLCB0aHVzIGhhdmluZyBELTEgKGluc3RlYWQgb2YgRClcbiAgLy8gc3RyaWRlcy5cbiAgY29uc3Qgc3RyaWRlcyA9IG5ldyBBcnJheShyYW5rIC0gMSk7XG4gIHN0cmlkZXNbcmFuayAtIDJdID0gc2hhcGVbcmFuayAtIDFdO1xuICBmb3IgKGxldCBpID0gcmFuayAtIDM7IGkgPj0gMDsgLS1pKSB7XG4gICAgc3RyaWRlc1tpXSA9IHN0cmlkZXNbaSArIDFdICogc2hhcGVbaSArIDFdO1xuICB9XG4gIHJldHVybiBzdHJpZGVzO1xufVxuXG5mdW5jdGlvbiBjcmVhdGVOZXN0ZWRBcnJheShcbiAgICBvZmZzZXQ6IG51bWJlciwgc2hhcGU6IG51bWJlcltdLCBhOiBUeXBlZEFycmF5LCBpc0NvbXBsZXggPSBmYWxzZSkge1xuICBjb25zdCByZXQgPSBuZXcgQXJyYXkoKTtcbiAgaWYgKHNoYXBlLmxlbmd0aCA9PT0gMSkge1xuICAgIGNvbnN0IGQgPSBzaGFwZVswXSAqIChpc0NvbXBsZXggPyAyIDogMSk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBkOyBpKyspIHtcbiAgICAgIHJldFtpXSA9IGFbb2Zmc2V0ICsgaV07XG4gICAgfVxuICB9IGVsc2Uge1xuICAgIGNvbnN0IGQgPSBzaGFwZVswXTtcbiAgICBjb25zdCByZXN0ID0gc2hhcGUuc2xpY2UoMSk7XG4gICAgY29uc3QgbGVuID0gcmVzdC5yZWR1Y2UoKGFjYywgYykgPT4gYWNjICogYykgKiAoaXNDb21wbGV4ID8gMiA6IDEpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgZDsgaSsrKSB7XG4gICAgICByZXRbaV0gPSBjcmVhdGVOZXN0ZWRBcnJheShvZmZzZXQgKyBpICogbGVuLCByZXN0LCBhLCBpc0NvbXBsZXgpO1xuICAgIH1cbiAgfVxuICByZXR1cm4gcmV0O1xufVxuXG4vLyBQcm92aWRlIGEgbmVzdGVkIGFycmF5IG9mIFR5cGVkQXJyYXkgaW4gZ2l2ZW4gc2hhcGUuXG5leHBvcnQgZnVuY3Rpb24gdG9OZXN0ZWRBcnJheShcbiAgICBzaGFwZTogbnVtYmVyW10sIGE6IFR5cGVkQXJyYXksIGlzQ29tcGxleCA9IGZhbHNlKSB7XG4gIGlmIChzaGFwZS5sZW5ndGggPT09IDApIHtcbiAgICAvLyBTY2FsYXIgdHlwZSBzaG91bGQgcmV0dXJuIGEgc2luZ2xlIG51bWJlci5cbiAgICByZXR1cm4gYVswXTtcbiAgfVxuICBjb25zdCBzaXplID0gc2hhcGUucmVkdWNlKChhY2MsIGMpID0+IGFjYyAqIGMpICogKGlzQ29tcGxleCA/IDIgOiAxKTtcbiAgaWYgKHNpemUgPT09IDApIHtcbiAgICAvLyBBIHRlbnNvciB3aXRoIHNoYXBlIHplcm8gc2hvdWxkIGJlIHR1cm5lZCBpbnRvIGVtcHR5IGxpc3QuXG4gICAgcmV0dXJuIFtdO1xuICB9XG4gIGlmIChzaXplICE9PSBhLmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcihgWyR7c2hhcGV9XSBkb2VzIG5vdCBtYXRjaCB0aGUgaW5wdXQgc2l6ZSAke2EubGVuZ3RofSR7XG4gICAgICAgIGlzQ29tcGxleCA/ICcgZm9yIGEgY29tcGxleCB0ZW5zb3InIDogJyd9LmApO1xuICB9XG5cbiAgcmV0dXJuIGNyZWF0ZU5lc3RlZEFycmF5KDAsIHNoYXBlLCBhLCBpc0NvbXBsZXgpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29udmVydEJhY2tlbmRWYWx1ZXNBbmRBcnJheUJ1ZmZlcihcbiAgICBkYXRhOiBCYWNrZW5kVmFsdWVzfEFycmF5QnVmZmVyLCBkdHlwZTogRGF0YVR5cGUpIHtcbiAgLy8gSWYgaXMgdHlwZSBVaW50OEFycmF5W10sIHJldHVybiBpdCBkaXJlY3RseS5cbiAgaWYgKEFycmF5LmlzQXJyYXkoZGF0YSkpIHtcbiAgICByZXR1cm4gZGF0YTtcbiAgfVxuICBpZiAoZHR5cGUgPT09ICdmbG9hdDMyJykge1xuICAgIHJldHVybiBkYXRhIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5ID8gZGF0YSA6IG5ldyBGbG9hdDMyQXJyYXkoZGF0YSk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicpIHtcbiAgICByZXR1cm4gZGF0YSBpbnN0YW5jZW9mIEludDMyQXJyYXkgPyBkYXRhIDogbmV3IEludDMyQXJyYXkoZGF0YSk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdib29sJyB8fCBkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICByZXR1cm4gVWludDhBcnJheS5mcm9tKG5ldyBJbnQzMkFycmF5KGRhdGEpKTtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVua25vd24gZHR5cGUgJHtkdHlwZX1gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gbWFrZU9uZXNUeXBlZEFycmF5PEQgZXh0ZW5kcyBEYXRhVHlwZT4oXG4gICAgc2l6ZTogbnVtYmVyLCBkdHlwZTogRCk6IERhdGFUeXBlTWFwW0RdIHtcbiAgY29uc3QgYXJyYXkgPSBtYWtlWmVyb3NUeXBlZEFycmF5KHNpemUsIGR0eXBlKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBhcnJheS5sZW5ndGg7IGkrKykge1xuICAgIGFycmF5W2ldID0gMTtcbiAgfVxuICByZXR1cm4gYXJyYXk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlWmVyb3NUeXBlZEFycmF5PEQgZXh0ZW5kcyBEYXRhVHlwZT4oXG4gICAgc2l6ZTogbnVtYmVyLCBkdHlwZTogRCk6IERhdGFUeXBlTWFwW0RdIHtcbiAgaWYgKGR0eXBlID09IG51bGwgfHwgZHR5cGUgPT09ICdmbG9hdDMyJyB8fCBkdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICByZXR1cm4gbmV3IEZsb2F0MzJBcnJheShzaXplKSBhcyBEYXRhVHlwZU1hcFtEXTtcbiAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ2ludDMyJykge1xuICAgIHJldHVybiBuZXcgSW50MzJBcnJheShzaXplKSBhcyBEYXRhVHlwZU1hcFtEXTtcbiAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ2Jvb2wnKSB7XG4gICAgcmV0dXJuIG5ldyBVaW50OEFycmF5KHNpemUpIGFzIERhdGFUeXBlTWFwW0RdO1xuICB9IGVsc2Uge1xuICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biBkYXRhIHR5cGUgJHtkdHlwZX1gKTtcbiAgfVxufVxuXG4vKipcbiAqIE1ha2UgbmVzdGVkIGBUeXBlZEFycmF5YCBmaWxsZWQgd2l0aCB6ZXJvcy5cbiAqIEBwYXJhbSBzaGFwZSBUaGUgc2hhcGUgaW5mb3JtYXRpb24gZm9yIHRoZSBuZXN0ZWQgYXJyYXkuXG4gKiBAcGFyYW0gZHR5cGUgZHR5cGUgb2YgdGhlIGFycmF5IGVsZW1lbnQuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBtYWtlWmVyb3NOZXN0ZWRUeXBlZEFycmF5PEQgZXh0ZW5kcyBEYXRhVHlwZT4oXG4gICAgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRCkge1xuICBjb25zdCBzaXplID0gc2hhcGUucmVkdWNlKChwcmV2LCBjdXJyKSA9PiBwcmV2ICogY3VyciwgMSk7XG4gIGlmIChkdHlwZSA9PSBudWxsIHx8IGR0eXBlID09PSAnZmxvYXQzMicpIHtcbiAgICByZXR1cm4gdG9OZXN0ZWRBcnJheShzaGFwZSwgbmV3IEZsb2F0MzJBcnJheShzaXplKSk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicpIHtcbiAgICByZXR1cm4gdG9OZXN0ZWRBcnJheShzaGFwZSwgbmV3IEludDMyQXJyYXkoc2l6ZSkpO1xuICB9IGVsc2UgaWYgKGR0eXBlID09PSAnYm9vbCcpIHtcbiAgICByZXR1cm4gdG9OZXN0ZWRBcnJheShzaGFwZSwgbmV3IFVpbnQ4QXJyYXkoc2l6ZSkpO1xuICB9IGVsc2Uge1xuICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biBkYXRhIHR5cGUgJHtkdHlwZX1gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0Tm9uTmVnYXRpdmVJbnRlZ2VyRGltZW5zaW9ucyhzaGFwZTogbnVtYmVyW10pIHtcbiAgc2hhcGUuZm9yRWFjaChkaW1TaXplID0+IHtcbiAgICBhc3NlcnQoXG4gICAgICAgIE51bWJlci5pc0ludGVnZXIoZGltU2l6ZSkgJiYgZGltU2l6ZSA+PSAwLFxuICAgICAgICAoKSA9PlxuICAgICAgICAgICAgYFRlbnNvciBtdXN0IGhhdmUgYSBzaGFwZSBjb21wcmlzZWQgb2YgcG9zaXRpdmUgaW50ZWdlcnMgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGBzaGFwZSBbJHtzaGFwZX1dLmApO1xuICB9KTtcbn1cblxuLyoqXG4gKiBDb21wdXRlcyBmbGF0IGluZGV4IGZvciBhIGdpdmVuIGxvY2F0aW9uIChtdWx0aWRpbWVudGlvbnNhbCBpbmRleCkgaW4gYVxuICogVGVuc29yL211bHRpZGltZW5zaW9uYWwgYXJyYXkuXG4gKlxuICogQHBhcmFtIGxvY3MgTG9jYXRpb24gaW4gdGhlIHRlbnNvci5cbiAqIEBwYXJhbSByYW5rIFJhbmsgb2YgdGhlIHRlbnNvci5cbiAqIEBwYXJhbSBzdHJpZGVzIFRlbnNvciBzdHJpZGVzLlxuICovXG5leHBvcnQgZnVuY3Rpb24gbG9jVG9JbmRleChcbiAgICBsb2NzOiBudW1iZXJbXSwgcmFuazogbnVtYmVyLCBzdHJpZGVzOiBudW1iZXJbXSk6IG51bWJlciB7XG4gIGlmIChyYW5rID09PSAwKSB7XG4gICAgcmV0dXJuIDA7XG4gIH0gZWxzZSBpZiAocmFuayA9PT0gMSkge1xuICAgIHJldHVybiBsb2NzWzBdO1xuICB9XG4gIGxldCBpbmRleCA9IGxvY3NbbG9jcy5sZW5ndGggLSAxXTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgIGluZGV4ICs9IHN0cmlkZXNbaV0gKiBsb2NzW2ldO1xuICB9XG4gIHJldHVybiBpbmRleDtcbn1cblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgbG9jYXRpb24gKG11bHRpZGltZW5zaW9uYWwgaW5kZXgpIGluIGFcbiAqIHRlbnNvci9tdWx0aWRpbWVudGlvbmFsIGFycmF5IGZvciBhIGdpdmVuIGZsYXQgaW5kZXguXG4gKlxuICogQHBhcmFtIGluZGV4IEluZGV4IGluIGZsYXQgYXJyYXkuXG4gKiBAcGFyYW0gcmFuayBSYW5rIG9mIHRlbnNvci5cbiAqIEBwYXJhbSBzdHJpZGVzIFN0cmlkZXMgb2YgdGVuc29yLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaW5kZXhUb0xvYyhcbiAgICBpbmRleDogbnVtYmVyLCByYW5rOiBudW1iZXIsIHN0cmlkZXM6IG51bWJlcltdKTogbnVtYmVyW10ge1xuICBpZiAocmFuayA9PT0gMCkge1xuICAgIHJldHVybiBbXTtcbiAgfSBlbHNlIGlmIChyYW5rID09PSAxKSB7XG4gICAgcmV0dXJuIFtpbmRleF07XG4gIH1cbiAgY29uc3QgbG9jczogbnVtYmVyW10gPSBuZXcgQXJyYXkocmFuayk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICBsb2NzW2ldID0gTWF0aC5mbG9vcihpbmRleCAvIHN0cmlkZXNbaV0pO1xuICAgIGluZGV4IC09IGxvY3NbaV0gKiBzdHJpZGVzW2ldO1xuICB9XG4gIGxvY3NbbG9jcy5sZW5ndGggLSAxXSA9IGluZGV4O1xuICByZXR1cm4gbG9jcztcbn1cblxuLyoqXG4gKiBUaGlzIG1ldGhvZCBhc3NlcnRzIHdoZXRoZXIgYW4gb2JqZWN0IGlzIGEgUHJvbWlzZSBpbnN0YW5jZS5cbiAqIEBwYXJhbSBvYmplY3RcbiAqL1xuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby1hbnlcbmV4cG9ydCBmdW5jdGlvbiBpc1Byb21pc2Uob2JqZWN0OiBhbnkpOiBvYmplY3QgaXMgUHJvbWlzZTx1bmtub3duPiB7XG4gIC8vICBXZSBjaG9zZSB0byBub3QgdXNlICdvYmogaW5zdGFuY2VPZiBQcm9taXNlJyBmb3IgdHdvIHJlYXNvbnM6XG4gIC8vICAxLiBJdCBvbmx5IHJlbGlhYmx5IHdvcmtzIGZvciBlczYgUHJvbWlzZSwgbm90IG90aGVyIFByb21pc2VcbiAgLy8gIGltcGxlbWVudGF0aW9ucy5cbiAgLy8gIDIuIEl0IGRvZXNuJ3Qgd29yayB3aXRoIGZyYW1ld29yayB0aGF0IHVzZXMgem9uZS5qcy4gem9uZS5qcyBtb25rZXlcbiAgLy8gIHBhdGNoIHRoZSBhc3luYyBjYWxscywgc28gaXQgaXMgcG9zc2libGUgdGhlIG9iaiAocGF0Y2hlZCkgaXNcbiAgLy8gIGNvbXBhcmluZyB0byBhIHByZS1wYXRjaGVkIFByb21pc2UuXG4gIHJldHVybiBvYmplY3QgJiYgb2JqZWN0LnRoZW4gJiYgdHlwZW9mIG9iamVjdC50aGVuID09PSAnZnVuY3Rpb24nO1xufVxuIl19