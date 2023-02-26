/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/**
 * TensorFlow.js Layers: Merge Layers.
 */
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy, util } from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import { Layer } from '../engine/topology';
import { NotImplementedError, ValueError } from '../errors';
import { l2Normalize } from '../losses';
import * as generic_utils from '../utils/generic_utils';
import * as mathUtils from '../utils/math_utils';
import { getExactlyOneShape } from '../utils/types_utils';
/**
 * Generic Merge layer for element-wise merge functions.
 *
 * Used to implement `Sum`, `Average`, `Concatenate`, etc.
 */
export class Merge extends Layer {
    constructor(args) {
        super(args || {});
        this.supportsMasking = true;
    }
    /**
     * Logic for merging multiple tensors, to be overridden by subclasses.
     * @param inputs
     */
    mergeFunction(inputs) {
        throw new NotImplementedError();
    }
    /**
     * Computes the shape of the result of an elementwise operation.
     *
     * @param shape1: Shape of the first tensor.
     * @param shape2: Shape of the second tensor.
     * @returns Expected output shape when an elementwise operation is carried
     *   out on 2 tensors with shapes `shape1` and `shape2`.
     * @throws ValueError: If `shape1` and `shape2` are not compatible for
     *   element-wise operations.
     */
    computeElementwiseOpOutputShape(shape1, shape2) {
        if (shape1 == null || shape2 == null) {
            return null;
        }
        else if (shape1.length < shape2.length) {
            return this.computeElementwiseOpOutputShape(shape2, shape1);
        }
        else if (shape2.length === 0) {
            return shape1;
        }
        const outputShape = shape1.slice(0, shape1.length - shape2.length);
        for (let k = 0; k < shape2.length; ++k) {
            const i = shape1[shape1.length - shape2.length + k];
            const j = shape2[k];
            if (i == null || j == null || i < 0 || j < 0) {
                outputShape.push(null);
            }
            else if (i === 1) {
                outputShape.push(j);
            }
            else if (j === 1) {
                outputShape.push(i);
            }
            else {
                if (i !== j) {
                    throw new ValueError('Operands could not be broadcast together with shapes ' +
                        JSON.stringify(shape1) + ' ' + JSON.stringify(shape2));
                }
                outputShape.push(i);
            }
        }
        return outputShape;
    }
    build(inputShape) {
        // Used purely for shape validation.
        if (Array.isArray(inputShape) && !Array.isArray(inputShape[0])) {
            // Make sure that inputShape is an Array of shape.
            inputShape = [getExactlyOneShape(inputShape)];
        }
        inputShape = inputShape;
        if (inputShape.length < 2) {
            throw new ValueError('A merge layer should be called on an Array of at least 2 inputs.' +
                ` Got ${inputShape.length} input(s).`);
        }
        // Make sure that there is at most one unique batch size among the input
        // shapes.
        let batchSizes = [];
        for (const shape of inputShape) {
            if (shape != null && shape[0] !== null) {
                batchSizes.push(shape[0]);
            }
        }
        batchSizes = generic_utils.unique(batchSizes);
        if (batchSizes.length > 1) {
            throw new ValueError(`Can not merge tensors with different batch sizes. ` +
                `Got tensors with shapes: ${JSON.stringify(inputShape)}.`);
        }
        let outputShape = inputShape[0] == null ? null : inputShape[0].slice(1);
        for (let i = 1; i < inputShape.length; ++i) {
            const shape = inputShape[i] == null ? null : inputShape[i].slice(1);
            outputShape = this.computeElementwiseOpOutputShape(outputShape, shape);
        }
        // If the inputs have different ranks, we have to reshape them to make them
        // broadcastable.
        const allRanks = inputShape.map(shape => shape.length);
        if (inputShape.indexOf(null) === -1 &&
            generic_utils.unique(allRanks).length === 1) {
            this.reshapeRequired = false;
        }
        else {
            this.reshapeRequired = true;
        }
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = inputs;
            if (this.reshapeRequired) {
                const reshapedInputs = [];
                const inputDims = inputs.map(input => input.rank);
                if (inputDims.indexOf(null) === -1) {
                    // If ranks of all inputs are available, we simply expand each of them
                    // at axis=1 until all of them have the same rank.
                    const maxNDim = mathUtils.max(inputDims);
                    for (let x of inputs) {
                        const xNDim = x.rank;
                        for (let k = 0; k < maxNDim - xNDim; ++k) {
                            x = K.expandDims(x, 1);
                        }
                        reshapedInputs.push(x);
                    }
                    return this.mergeFunction(reshapedInputs);
                }
                else {
                    // Transpose all inputs so that batch size is the last dimension.
                    // [batchSize, dim1, dim2, ...] -> [dim1, dim2, ..., batchSize]
                    let transposed = false;
                    for (const x of inputs) {
                        const xNDim = x.rank;
                        if (xNDim == null) {
                            const xShape = x.shape;
                            const batchSize = xShape[0];
                            const newShape = xShape.slice(1).concat([batchSize]);
                            let xTransposed = tfc.reshape(x, [batchSize].concat(mathUtils.arrayProd(xShape.slice(1))));
                            xTransposed = tfc.transpose(xTransposed, [1, 0]);
                            xTransposed = tfc.reshape(xTransposed, newShape);
                            reshapedInputs.push(xTransposed);
                            transposed = true;
                        }
                        else if (xNDim > 1) {
                            const dims = mathUtils.range(1, xNDim).concat([0]);
                            reshapedInputs.push(tfc.transpose(x, dims));
                            transposed = true;
                        }
                        else {
                            // We don't transpose inputs if they are 1D vectors or scalars.
                            reshapedInputs.push(x);
                        }
                    }
                    let y = this.mergeFunction(reshapedInputs);
                    const yNDim = y.rank;
                    if (transposed) {
                        // If inputs have been transposed, we have to transpose the output
                        // too.
                        if (yNDim == null) {
                            const yShape = y.shape;
                            const yNDim = yShape.length;
                            const batchSize = yShape[yNDim - 1];
                            const newShape = [batchSize].concat(yShape.slice(0, yShape.length - 1));
                            y = tfc.reshape(tfc.transpose(tfc.reshape(y, [-1, batchSize]), [1, 0]), newShape);
                        }
                        else if (yNDim > 1) {
                            const dims = [yNDim - 1].concat(mathUtils.range(0, yNDim - 1));
                            y = tfc.transpose(y, dims);
                        }
                    }
                    return y;
                }
            }
            else {
                return this.mergeFunction(inputs);
            }
        });
    }
    computeOutputShape(inputShape) {
        inputShape = inputShape;
        let outputShape;
        if (inputShape[0] == null) {
            outputShape = null;
        }
        else {
            outputShape = inputShape[0].slice(1);
        }
        for (let i = 1; i < inputShape.length; ++i) {
            const shape = inputShape[i] == null ? null : inputShape[i].slice(1);
            outputShape = this.computeElementwiseOpOutputShape(outputShape, shape);
        }
        let batchSizes = [];
        for (const shape of inputShape) {
            if (shape != null && shape[0] !== null) {
                batchSizes.push(shape[0]);
            }
        }
        batchSizes = generic_utils.unique(batchSizes);
        if (batchSizes.length === 1) {
            outputShape = batchSizes.concat(outputShape);
        }
        else {
            outputShape = [null].concat(outputShape);
        }
        return outputShape;
    }
    computeMask(inputs, mask) {
        return tfc.tidy(() => {
            if (mask == null) {
                return null;
            }
            if (!Array.isArray(mask)) {
                throw new ValueError('`mask` should be an Array');
            }
            if (!Array.isArray(inputs)) {
                throw new ValueError('`inputs` should be an Array');
            }
            if (mask.length !== inputs.length) {
                throw new ValueError(`The Array 'inputs' and 'mask' are expected to have the same ` +
                    `length, but have different lengths ` +
                    `(${inputs.length} vs ${mask.length})`);
            }
            if (mask.every(m => m == null)) {
                return null;
            }
            mask = mask.map(m => m == null ? m : tfc.expandDims(m, 0));
            let output = mask[0];
            for (let i = 1; i < mask.length - 1; ++i) {
                output = tfc.logicalAnd(output, mask[i]);
            }
            return output;
        });
    }
}
export class Add extends Merge {
    constructor(args) {
        super(args);
    }
    mergeFunction(inputs) {
        return tidy(() => {
            let output = inputs[0].clone();
            for (let i = 1; i < inputs.length; ++i) {
                output = tfc.add(output, inputs[i]);
            }
            return output;
        });
    }
}
/** @nocollapse */
Add.className = 'Add';
serialization.registerClass(Add);
/**
 * Calculate the element-wise sum of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Add` layer, by using no input argument
 *    or a single configuration argument. The resultant `Add` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const addLayer = tf.layers.add();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = addLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.add([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const input2 = tf.tensor2d([10, 20, 30, 40], [2, 2]);
 * tf.layers.add([input1, input2]).print();
 * // Gives [[11, 22], [33, 44]].
 *
 */
export function add(config) {
    if (Array.isArray(config)) {
        const layer = new Add({});
        return layer.apply(config);
    }
    else {
        return new Add(config);
    }
}
export class Multiply extends Merge {
    constructor(args) {
        super(args);
    }
    mergeFunction(inputs) {
        return tidy(() => {
            let output = inputs[0].clone();
            for (let i = 1; i < inputs.length; ++i) {
                output = tfc.mul(output, inputs[i]);
            }
            return output;
        });
    }
}
/** @nocollapse */
Multiply.className = 'Multiply';
serialization.registerClass(Multiply);
/**
 * Calculate the element-wise product of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Multiply` layer, by using no input argument
 *    or a single configuration argument. The resultant `Multiply` layer can
 *    then be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const multiplyLayer = tf.layers.multiply();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = multiplyLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.multiply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const input2 = tf.tensor2d([10, 20, 30, 40], [2, 2]);
 * tf.layers.multiply([input1, input2]).print();
 * // Gives [[10, 40], [90, 160]].
 *
 */
export function multiply(config) {
    if (Array.isArray(config)) {
        const layer = new Multiply({});
        return layer.apply(config);
    }
    else {
        return new Multiply(config);
    }
}
export class Average extends Merge {
    constructor(args) {
        super(args);
    }
    mergeFunction(inputs) {
        return tidy(() => {
            let output = inputs[0].clone();
            for (let i = 1; i < inputs.length; ++i) {
                output = tfc.add(output, inputs[i]);
            }
            return tfc.mul(1 / inputs.length, output);
        });
    }
}
/** @nocollapse */
Average.className = 'Average';
serialization.registerClass(Average);
/**
 * Calculate the element-wise arithmetic mean of inputs, which all have the same
 * shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Average` layer, by using no input argument
 *    or a single configuration argument. The resultant `Average` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const averageLayer = tf.layers.average();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = averageLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.average([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const input2 = tf.tensor2d([10, 20, 30, 40], [2, 2]);
 * tf.layers.average([input1, input2]).print();
 * // Gives [[5.5, 11], [16.5, 22]].
 *
 */
export function average(config) {
    if (Array.isArray(config)) {
        const layer = new Average({});
        return layer.apply(config);
    }
    else {
        return new Average(config);
    }
}
export class Maximum extends Merge {
    constructor(args) {
        super(args);
    }
    mergeFunction(inputs) {
        return tidy(() => {
            let output = inputs[0];
            for (let i = 1; i < inputs.length; ++i) {
                output = tfc.maximum(output, inputs[i]);
            }
            return output;
        });
    }
}
/** @nocollapse */
Maximum.className = 'Maximum';
serialization.registerClass(Maximum);
/**
 * Calculate the element-wise maximum of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Maximum` layer, by using no input argument
 *    or a single configuration argument. The resultant `Maximum` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const maximumLayer = tf.layers.maximum();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = maximumLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.maximum([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 20, 3, 40], [2, 2]);
 * const input2 = tf.tensor2d([10, 2, 30, 4], [2, 2]);
 * tf.layers.maximum([input1, input2]).print();
 * // Gives [[10, 20], [30, 40]].
 *
 */
export function maximum(config) {
    if (Array.isArray(config)) {
        const layer = new Maximum({});
        return layer.apply(config);
    }
    else {
        return new Maximum(config);
    }
}
export class Minimum extends Merge {
    constructor(args) {
        super(args);
    }
    mergeFunction(inputs) {
        return tidy(() => {
            let output = inputs[0];
            for (let i = 1; i < inputs.length; ++i) {
                output = tfc.minimum(output, inputs[i]);
            }
            return output;
        });
    }
}
/** @nocollapse */
Minimum.className = 'Minimum';
serialization.registerClass(Minimum);
/**
 * Calculate the element-wise minimum of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Minimum` layer, by using no input argument
 *    or a single configuration argument. The resultant `Minimum` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const minimumLayer = tf.layers.minimum();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = minimumLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.minimum([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 20, 3, 40], [2, 2]);
 * const input2 = tf.tensor2d([10, 2, 30, 4], [2, 2]);
 * tf.layers.minimum([input1, input2]).print();
 * // Gives [[1, 2], [3, 4]].
 *
 */
export function minimum(config) {
    if (Array.isArray(config)) {
        const layer = new Minimum({});
        return layer.apply(config);
    }
    else {
        return new Minimum(config);
    }
}
export class Concatenate extends Merge {
    constructor(args) {
        super(args);
        this.DEFAULT_AXIS = -1;
        if (args == null) {
            args = {};
        }
        this.axis = args.axis == null ? this.DEFAULT_AXIS : args.axis;
        this.supportsMasking = true;
        this.reshapeRequired = false;
    }
    build(inputShape) {
        // Used purely for shape validation.]
        if (!(Array.isArray(inputShape) && Array.isArray(inputShape[0])) ||
            inputShape.length === 1) {
            throw new ValueError('A `Concatenate` layer should be called on a list of at least 2 ' +
                'inputs');
        }
        inputShape = inputShape;
        let allNoneShape = true;
        for (const shape of inputShape) {
            if (shape != null) {
                allNoneShape = false;
                break;
            }
        }
        if (allNoneShape) {
            return;
        }
        const shapeSet = [];
        for (let i = 0; i < inputShape.length; ++i) {
            const shapeWithoutConcatAxis = inputShape[i].slice();
            shapeWithoutConcatAxis.splice(this.axis, 1);
            let exists = false;
            for (const shape of shapeSet) {
                if (util.arraysEqual(shape, shapeWithoutConcatAxis)) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                shapeSet.push(shapeWithoutConcatAxis);
            }
        }
        if (shapeSet.length > 1) {
            throw new ValueError('A `Concatenate` layer requires inputs with matching shapes ' +
                'except for the concat axis. Got input shapes: ' +
                JSON.stringify(inputShape));
        }
    }
    mergeFunction(inputs) {
        return tidy(() => {
            return K.concatenate(inputs, this.axis);
        });
    }
    computeOutputShape(inputShape) {
        if (!(Array.isArray(inputShape) && Array.isArray(inputShape[0]))) {
            throw new ValueError('A `Concatenate` layer should be called on a list of inputs.');
        }
        const inputShapes = inputShape;
        const outputShape = inputShapes[0].slice();
        const axis = this.axis < 0 ? outputShape.length + this.axis : this.axis;
        // Porting Note: the line above is because TypeScript doesn't support
        //   negative indices.
        for (const shape of inputShapes.slice(1)) {
            if (outputShape[axis] == null || shape[axis] == null) {
                outputShape[axis] = null;
                break;
            }
            outputShape[axis] += shape[axis];
        }
        return outputShape;
    }
    computeMask(inputs, mask) {
        if (mask == null) {
            return null;
        }
        if (!Array.isArray(mask)) {
            throw new ValueError('`mask` should be an array for Concatenate');
        }
        if (!Array.isArray(inputs)) {
            throw new ValueError('`inputs` should be an array for Concatenate');
        }
        if (mask.length !== inputs.length) {
            throw new ValueError(`Mismatch in the length of mask (${mask.length}) ` +
                `and the legnth of inputs (${inputs.length})`);
        }
        return tfc.tidy(() => {
            let allNullMasks = true;
            mask.forEach(m => {
                if (m != null) {
                    allNullMasks = false;
                    return;
                }
            });
            if (allNullMasks) {
                return null;
            }
            const outputMasks = [];
            for (let i = 0; i < inputs.length; ++i) {
                if (mask[i] == null) {
                    // Input is unmasked. Append all 1's to masks.
                    outputMasks.push(tfc.cast(tfc.onesLike(inputs[i]), 'bool'));
                }
                else if (mask[i].rank < inputs[i].rank) {
                    // Mask is smaller than the input, expand it.
                    outputMasks.push(tfc.expandDims(mask[i], -1));
                }
                else {
                    outputMasks.push(mask[i]);
                }
            }
            const concatenatedMasks = tfc.concat(outputMasks, this.axis);
            return tfc.all(concatenatedMasks, -1, false);
        });
    }
    getConfig() {
        const config = {
            'axis': this.axis,
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
Concatenate.className = 'Concatenate';
serialization.registerClass(Concatenate);
/**
 * Concatenate an `Array` of inputs.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Concatenate` layer, by using no input argument
 *    or a single configuration argument. The resultant `Concatenate` layer can
 *    then be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const concatLayer = tf.layers.concatenate();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 3]});
 * const input2 = tf.input({shape: [2, 4]});
 * const output = concatLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 7], with the first dimension as the undetermined batch
 * // dimension and the last dimension as the result of concatenating the
 * // last dimensions of the two inputs.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 3]});
 * const input2 = tf.input({shape: [2, 4]});
 * const output = tf.layers.concatenate([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension and the last dimension as the result of concatenating the
 * // last dimensions of the two inputs.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
 * const input2 = tf.tensor2d([[10, 20], [30, 40]], [2, 2]);
 * tf.layers.concatenate([input1, input2]).print();
 * // Gives [[1, 2, 10, 20], [3, 4, 30, 40]].
 *
 */
export function concatenate(config) {
    if (Array.isArray(config)) {
        const layer = new Concatenate({});
        return layer.apply(config);
    }
    else {
        return new Concatenate(config);
    }
}
/**
 * Interpretable potentially negative axis index.
 *
 * For example, given axis = -1, and dim = 3, this function will return 2.
 *
 * @param axis The axis index, may be a positive, zero or negative integer.
 * @param dim Total number of dimensions, a positive integer.
 * @returns A non-negative axis index equivalent to the input `axis`.
 */
function interpretAxis(axis, dim) {
    while (axis < 0) {
        axis += dim;
    }
    return axis;
}
function batchDot(x, y, axes) {
    if (x.shape.length > 3 || y.shape.length > 3) {
        throw new NotImplementedError('batchDot is not implemented for tensors of 4D or higher rank yet');
    }
    tfc.util.assert(x.shape.length >= 2, () => `batchDot requires the rank of x to be >= 2, ` +
        `but got ${x.shape.length}`);
    tfc.util.assert(x.shape.length >= 2, () => `batchDot requires the rank of y to be >= 2, ` +
        `but got ${y.shape.length}`);
    if (typeof axes === 'number') {
        axes = [axes, axes];
    }
    if (x.dtype === 'complex64' || y.dtype === 'complex64') {
        throw new NotImplementedError('batchDot is not implemented for complex64-type Tensors yet.');
    }
    const xNDim = x.shape.length;
    const yNDim = y.shape.length;
    if (axes == null) {
        // Behave like batchMatmul by default.
        axes = [xNDim - 1, yNDim - 2];
    }
    const axesArray = axes;
    return tfc.tidy(() => {
        let diff;
        if (xNDim > yNDim) {
            diff = xNDim - yNDim;
            const diffShape = [];
            for (let i = 0; i < diff; ++i) {
                diffShape.push(1);
            }
            y = tfc.reshape(y, y.shape.concat(diffShape));
        }
        else if (yNDim > xNDim) {
            diff = yNDim - xNDim;
            const diffShape = [];
            for (let i = 0; i < diff; ++i) {
                diffShape.push(1);
            }
            x = tfc.reshape(x, x.shape.concat(diffShape));
        }
        else {
            diff = 0;
        }
        let out;
        if (x.shape.length === 2 && y.shape.length === 2) {
            if (axesArray[0] === axesArray[1]) {
                out = tfc.sum(tfc.mul(x, y), axesArray[0]);
            }
            else {
                out = tfc.sum(tfc.mul(tfc.transpose(x, [1, 0]), y), axesArray[1]);
            }
        }
        else {
            const adjX = axesArray[0] !== x.shape.length - 1;
            const adjY = axesArray[1] === y.shape.length - 1;
            out = tfc.matMul(x, y, adjX, adjY);
        }
        if (diff > 0) {
            let idx;
            if (xNDim > yNDim) {
                idx = xNDim + yNDim - 3;
            }
            else {
                idx = xNDim - 1;
            }
            const squeezeAxes = [];
            for (let i = idx; i < idx + diff; ++i) {
                squeezeAxes.push(i);
            }
            out = tfc.squeeze(out, squeezeAxes);
        }
        if (out.shape.length === 1) {
            out = tfc.expandDims(out, 1);
        }
        return out;
    });
}
export class Dot extends Merge {
    constructor(args) {
        super(args);
        this.axes = args.axes;
        this.normalize = args.normalize == null ? false : args.normalize;
        this.supportsMasking = true;
        this.reshapeRequired = false;
    }
    build(inputShape) {
        tfc.util.assert(Array.isArray(inputShape) && inputShape.length === 2 &&
            Array.isArray(inputShape[0]) && Array.isArray(inputShape[1]), () => 'A `Dot` layer should be called on a list of exactly 2 inputs.');
        const shape1 = inputShape[0];
        const shape2 = inputShape[1];
        if (shape1.length > 3 || shape2.length > 3) {
            throw new NotImplementedError('Dot layer does not support tensors of 4D or higher rank yet.');
        }
        const axes = this.interpretAxes(shape1, shape2);
        if (shape1[axes[0]] !== shape2[axes[1]]) {
            throw new ValueError(`Dimension incompatibility: ` +
                `${shape1[axes[0]]} !== ${shape2[axes[1]]}`);
        }
    }
    mergeFunction(inputs) {
        if (inputs.length !== 2) {
            throw new ValueError('A `Dot` layer must be called on exactly 2 inputs, ' +
                `but received ${inputs.length} input(s).`);
        }
        let x1 = inputs[0];
        let x2 = inputs[1];
        let axes;
        if (!Array.isArray(this.axes)) {
            axes = [
                interpretAxis(this.axes, x1.shape.length),
                interpretAxis(this.axes, x2.shape.length)
            ];
        }
        else {
            axes = this.axes.map((axis, i) => interpretAxis(axis, inputs[i].shape.length));
        }
        if (this.normalize) {
            x1 = l2Normalize(x1, axes[0]);
            x2 = l2Normalize(x2, axes[1]);
        }
        return batchDot(x1, x2, axes);
    }
    interpretAxes(shape1, shape2) {
        let axes;
        if (!Array.isArray(this.axes)) {
            // `this.axes` is a single integer.
            axes = [
                interpretAxis(this.axes, shape1.length),
                interpretAxis(this.axes, shape2.length)
            ];
        }
        else {
            // `this.axes` is an Array of integers.
            axes = this.axes;
        }
        return axes;
    }
    computeOutputShape(inputShape) {
        tfc.util.assert(Array.isArray(inputShape) && inputShape.length === 2 &&
            Array.isArray(inputShape[0]) && Array.isArray(inputShape[1]), () => 'A `Dot` layer should be called on a list of exactly 2 inputs.');
        const shape1 = inputShape[0].slice();
        const shape2 = inputShape[1].slice();
        if (shape1.length > 3 || shape2.length > 3) {
            throw new NotImplementedError('Dot layer does not support tensors of 4D or higher rank yet.');
        }
        const axes = this.interpretAxes(shape1, shape2);
        shape1.splice(axes[0], 1);
        shape2.splice(axes[1], 1);
        shape2.splice(0, 1);
        const outputShape = shape1.concat(shape2);
        if (outputShape.length === 1) {
            outputShape.push(1);
        }
        return outputShape;
    }
    computeMask(inputs, mask) {
        return null;
    }
    getConfig() {
        const config = {
            'axes': this.axes,
            'normalize': this.normalize
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
Dot.className = 'Dot';
serialization.registerClass(Dot);
// TODO(cais): Add functional interfaces for the merge layers.
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWVyZ2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvbGF5ZXJzL21lcmdlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUg7O0dBRUc7QUFFSCxPQUFPLEtBQUssR0FBRyxNQUFNLHVCQUF1QixDQUFDO0FBQzdDLE9BQU8sRUFBQyxhQUFhLEVBQVUsSUFBSSxFQUFFLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ3hFLE9BQU8sS0FBSyxDQUFDLE1BQU0seUJBQXlCLENBQUM7QUFDN0MsT0FBTyxFQUFDLEtBQUssRUFBNEIsTUFBTSxvQkFBb0IsQ0FBQztBQUNwRSxPQUFPLEVBQUMsbUJBQW1CLEVBQUUsVUFBVSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBRTFELE9BQU8sRUFBQyxXQUFXLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFFdEMsT0FBTyxLQUFLLGFBQWEsTUFBTSx3QkFBd0IsQ0FBQztBQUN4RCxPQUFPLEtBQUssU0FBUyxNQUFNLHFCQUFxQixDQUFDO0FBQ2pELE9BQU8sRUFBQyxrQkFBa0IsRUFBQyxNQUFNLHNCQUFzQixDQUFDO0FBRXhEOzs7O0dBSUc7QUFDSCxNQUFNLE9BQWdCLEtBQU0sU0FBUSxLQUFLO0lBR3ZDLFlBQVksSUFBZ0I7UUFDMUIsS0FBSyxDQUFDLElBQUksSUFBSSxFQUFFLENBQUMsQ0FBQztRQUNsQixJQUFJLENBQUMsZUFBZSxHQUFHLElBQUksQ0FBQztJQUM5QixDQUFDO0lBRUQ7OztPQUdHO0lBQ08sYUFBYSxDQUFDLE1BQWdCO1FBQ3RDLE1BQU0sSUFBSSxtQkFBbUIsRUFBRSxDQUFDO0lBQ2xDLENBQUM7SUFFRDs7Ozs7Ozs7O09BU0c7SUFDSywrQkFBK0IsQ0FBQyxNQUFhLEVBQUUsTUFBYTtRQUNsRSxJQUFJLE1BQU0sSUFBSSxJQUFJLElBQUksTUFBTSxJQUFJLElBQUksRUFBRTtZQUNwQyxPQUFPLElB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TENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qKlxuICogVGVuc29yRmxvdy5qcyBMYXllcnM6IE1lcmdlIExheWVycy5cbiAqL1xuXG5pbXBvcnQgKiBhcyB0ZmMgZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7c2VyaWFsaXphdGlvbiwgVGVuc29yLCB0aWR5LCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0ICogYXMgSyBmcm9tICcuLi9iYWNrZW5kL3RmanNfYmFja2VuZCc7XG5pbXBvcnQge0xheWVyLCBMYXllckFyZ3MsIFN5bWJvbGljVGVuc29yfSBmcm9tICcuLi9lbmdpbmUvdG9wb2xvZ3knO1xuaW1wb3J0IHtOb3RJbXBsZW1lbnRlZEVycm9yLCBWYWx1ZUVycm9yfSBmcm9tICcuLi9lcnJvcnMnO1xuaW1wb3J0IHtTaGFwZX0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L2NvbW1vbic7XG5pbXBvcnQge2wyTm9ybWFsaXplfSBmcm9tICcuLi9sb3NzZXMnO1xuaW1wb3J0IHtLd2FyZ3N9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIGdlbmVyaWNfdXRpbHMgZnJvbSAnLi4vdXRpbHMvZ2VuZXJpY191dGlscyc7XG5pbXBvcnQgKiBhcyBtYXRoVXRpbHMgZnJvbSAnLi4vdXRpbHMvbWF0aF91dGlscyc7XG5pbXBvcnQge2dldEV4YWN0bHlPbmVTaGFwZX0gZnJvbSAnLi4vdXRpbHMvdHlwZXNfdXRpbHMnO1xuXG4vKipcbiAqIEdlbmVyaWMgTWVyZ2UgbGF5ZXIgZm9yIGVsZW1lbnQtd2lzZSBtZXJnZSBmdW5jdGlvbnMuXG4gKlxuICogVXNlZCB0byBpbXBsZW1lbnQgYFN1bWAsIGBBdmVyYWdlYCwgYENvbmNhdGVuYXRlYCwgZXRjLlxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgTWVyZ2UgZXh0ZW5kcyBMYXllciB7XG4gIHByb3RlY3RlZCByZXNoYXBlUmVxdWlyZWQ6IGJvb2xlYW47XG5cbiAgY29uc3RydWN0b3IoYXJncz86IExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MgfHwge30pO1xuICAgIHRoaXMuc3VwcG9ydHNNYXNraW5nID0gdHJ1ZTtcbiAgfVxuXG4gIC8qKlxuICAgKiBMb2dpYyBmb3IgbWVyZ2luZyBtdWx0aXBsZSB0ZW5zb3JzLCB0byBiZSBvdmVycmlkZGVuIGJ5IHN1YmNsYXNzZXMuXG4gICAqIEBwYXJhbSBpbnB1dHNcbiAgICovXG4gIHByb3RlY3RlZCBtZXJnZUZ1bmN0aW9uKGlucHV0czogVGVuc29yW10pOiBUZW5zb3Ige1xuICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKCk7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIHNoYXBlIG9mIHRoZSByZXN1bHQgb2YgYW4gZWxlbWVudHdpc2Ugb3BlcmF0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0gc2hhcGUxOiBTaGFwZSBvZiB0aGUgZmlyc3QgdGVuc29yLlxuICAgKiBAcGFyYW0gc2hhcGUyOiBTaGFwZSBvZiB0aGUgc2Vjb25kIHRlbnNvci5cbiAgICogQHJldHVybnMgRXhwZWN0ZWQgb3V0cHV0IHNoYXBlIHdoZW4gYW4gZWxlbWVudHdpc2Ugb3BlcmF0aW9uIGlzIGNhcnJpZWRcbiAgICogICBvdXQgb24gMiB0ZW5zb3JzIHdpdGggc2hhcGVzIGBzaGFwZTFgIGFuZCBgc2hhcGUyYC5cbiAgICogQHRocm93cyBWYWx1ZUVycm9yOiBJZiBgc2hhcGUxYCBhbmQgYHNoYXBlMmAgYXJlIG5vdCBjb21wYXRpYmxlIGZvclxuICAgKiAgIGVsZW1lbnQtd2lzZSBvcGVyYXRpb25zLlxuICAgKi9cbiAgcHJpdmF0ZSBjb21wdXRlRWxlbWVudHdpc2VPcE91dHB1dFNoYXBlKHNoYXBlMTogU2hhcGUsIHNoYXBlMjogU2hhcGUpOiBTaGFwZSB7XG4gICAgaWYgKHNoYXBlMSA9PSBudWxsIHx8IHNoYXBlMiA9PSBudWxsKSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9IGVsc2UgaWYgKHNoYXBlMS5sZW5ndGggPCBzaGFwZTIubGVuZ3RoKSB7XG4gICAgICByZXR1cm4gdGhpcy5jb21wdXRlRWxlbWVudHdpc2VPcE91dHB1dFNoYXBlKHNoYXBlMiwgc2hhcGUxKTtcbiAgICB9IGVsc2UgaWYgKHNoYXBlMi5sZW5ndGggPT09IDApIHtcbiAgICAgIHJldHVybiBzaGFwZTE7XG4gICAgfVxuICAgIGNvbnN0IG91dHB1dFNoYXBlOiBTaGFwZSA9IHNoYXBlMS5zbGljZSgwLCBzaGFwZTEubGVuZ3RoIC0gc2hhcGUyLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgayA9IDA7IGsgPCBzaGFwZTIubGVuZ3RoOyArK2spIHtcbiAgICAgIGNvbnN0IGkgPSBzaGFwZTFbc2hhcGUxLmxlbmd0aCAtIHNoYXBlMi5sZW5ndGggKyBrXTtcbiAgICAgIGNvbnN0IGogPSBzaGFwZTJba107XG4gICAgICBpZiAoaSA9PSBudWxsIHx8IGogPT0gbnVsbCB8fCBpIDwgMCB8fCBqIDwgMCkge1xuICAgICAgICBvdXRwdXRTaGFwZS5wdXNoKG51bGwpO1xuICAgICAgfSBlbHNlIGlmIChpID09PSAxKSB7XG4gICAgICAgIG91dHB1dFNoYXBlLnB1c2goaik7XG4gICAgICB9IGVsc2UgaWYgKGogPT09IDEpIHtcbiAgICAgICAgb3V0cHV0U2hhcGUucHVzaChpKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGlmIChpICE9PSBqKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgICdPcGVyYW5kcyBjb3VsZCBub3QgYmUgYnJvYWRjYXN0IHRvZ2V0aGVyIHdpdGggc2hhcGVzICcgK1xuICAgICAgICAgICAgICBKU09OLnN0cmluZ2lmeShzaGFwZTEpICsgJyAnICsgSlNPTi5zdHJpbmdpZnkoc2hhcGUyKSk7XG4gICAgICAgIH1cbiAgICAgICAgb3V0cHV0U2hhcGUucHVzaChpKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG91dHB1dFNoYXBlO1xuICB9XG5cbiAgb3ZlcnJpZGUgYnVpbGQoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IHZvaWQge1xuICAgIC8vIFVzZWQgcHVyZWx5IGZvciBzaGFwZSB2YWxpZGF0aW9uLlxuICAgIGlmIChBcnJheS5pc0FycmF5KGlucHV0U2hhcGUpICYmICFBcnJheS5pc0FycmF5KGlucHV0U2hhcGVbMF0pKSB7XG4gICAgICAvLyBNYWtlIHN1cmUgdGhhdCBpbnB1dFNoYXBlIGlzIGFuIEFycmF5IG9mIHNoYXBlLlxuICAgICAgaW5wdXRTaGFwZSA9IFtnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSldO1xuICAgIH1cbiAgICBpbnB1dFNoYXBlID0gaW5wdXRTaGFwZSBhcyBTaGFwZVtdO1xuICAgIGlmIChpbnB1dFNoYXBlLmxlbmd0aCA8IDIpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdBIG1lcmdlIGxheWVyIHNob3VsZCBiZSBjYWxsZWQgb24gYW4gQXJyYXkgb2YgYXQgbGVhc3QgMiBpbnB1dHMuJyArXG4gICAgICAgICAgYCBHb3QgJHtpbnB1dFNoYXBlLmxlbmd0aH0gaW5wdXQocykuYCk7XG4gICAgfVxuXG4gICAgLy8gTWFrZSBzdXJlIHRoYXQgdGhlcmUgaXMgYXQgbW9zdCBvbmUgdW5pcXVlIGJhdGNoIHNpemUgYW1vbmcgdGhlIGlucHV0XG4gICAgLy8gc2hhcGVzLlxuICAgIGxldCBiYXRjaFNpemVzOiBudW1iZXJbXSA9IFtdO1xuICAgIGZvciAoY29uc3Qgc2hhcGUgb2YgaW5wdXRTaGFwZSkge1xuICAgICAgaWYgKHNoYXBlICE9IG51bGwgJiYgc2hhcGVbMF0gIT09IG51bGwpIHtcbiAgICAgICAgYmF0Y2hTaXplcy5wdXNoKHNoYXBlWzBdKTtcbiAgICAgIH1cbiAgICB9XG4gICAgYmF0Y2hTaXplcyA9IGdlbmVyaWNfdXRpbHMudW5pcXVlKGJhdGNoU2l6ZXMpO1xuICAgIGlmIChiYXRjaFNpemVzLmxlbmd0aCA+IDEpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBDYW4gbm90IG1lcmdlIHRlbnNvcnMgd2l0aCBkaWZmZXJlbnQgYmF0Y2ggc2l6ZXMuIGAgK1xuICAgICAgICAgIGBHb3QgdGVuc29ycyB3aXRoIHNoYXBlczogJHtKU09OLnN0cmluZ2lmeShpbnB1dFNoYXBlKX0uYCk7XG4gICAgfVxuXG4gICAgbGV0IG91dHB1dFNoYXBlOiBTaGFwZSA9XG4gICAgICAgIGlucHV0U2hhcGVbMF0gPT0gbnVsbCA/IG51bGwgOiBpbnB1dFNoYXBlWzBdLnNsaWNlKDEpO1xuICAgIGZvciAobGV0IGkgPSAxOyBpIDwgaW5wdXRTaGFwZS5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3Qgc2hhcGUgPSBpbnB1dFNoYXBlW2ldID09IG51bGwgPyBudWxsIDogaW5wdXRTaGFwZVtpXS5zbGljZSgxKTtcbiAgICAgIG91dHB1dFNoYXBlID0gdGhpcy5jb21wdXRlRWxlbWVudHdpc2VPcE91dHB1dFNoYXBlKG91dHB1dFNoYXBlLCBzaGFwZSk7XG4gICAgfVxuICAgIC8vIElmIHRoZSBpbnB1dHMgaGF2ZSBkaWZmZXJlbnQgcmFua3MsIHdlIGhhdmUgdG8gcmVzaGFwZSB0aGVtIHRvIG1ha2UgdGhlbVxuICAgIC8vIGJyb2FkY2FzdGFibGUuXG4gICAgY29uc3QgYWxsUmFua3MgPSBpbnB1dFNoYXBlLm1hcChzaGFwZSA9PiBzaGFwZS5sZW5ndGgpO1xuICAgIGlmIChpbnB1dFNoYXBlLmluZGV4T2YobnVsbCkgPT09IC0xICYmXG4gICAgICAgIGdlbmVyaWNfdXRpbHMudW5pcXVlKGFsbFJhbmtzKS5sZW5ndGggPT09IDEpIHtcbiAgICAgIHRoaXMucmVzaGFwZVJlcXVpcmVkID0gZmFsc2U7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMucmVzaGFwZVJlcXVpcmVkID0gdHJ1ZTtcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgaW5wdXRzID0gaW5wdXRzIGFzIFRlbnNvcltdO1xuICAgICAgaWYgKHRoaXMucmVzaGFwZVJlcXVpcmVkKSB7XG4gICAgICAgIGNvbnN0IHJlc2hhcGVkSW5wdXRzOiBUZW5zb3JbXSA9IFtdO1xuICAgICAgICBjb25zdCBpbnB1dERpbXMgPSBpbnB1dHMubWFwKGlucHV0ID0+IGlucHV0LnJhbmspO1xuICAgICAgICBpZiAoaW5wdXREaW1zLmluZGV4T2YobnVsbCkgPT09IC0xKSB7XG4gICAgICAgICAgLy8gSWYgcmFua3Mgb2YgYWxsIGlucHV0cyBhcmUgYXZhaWxhYmxlLCB3ZSBzaW1wbHkgZXhwYW5kIGVhY2ggb2YgdGhlbVxuICAgICAgICAgIC8vIGF0IGF4aXM9MSB1bnRpbCBhbGwgb2YgdGhlbSBoYXZlIHRoZSBzYW1lIHJhbmsuXG4gICAgICAgICAgY29uc3QgbWF4TkRpbSA9IG1hdGhVdGlscy5tYXgoaW5wdXREaW1zKTtcbiAgICAgICAgICBmb3IgKGxldCB4IG9mIGlucHV0cykge1xuICAgICAgICAgICAgY29uc3QgeE5EaW0gPSB4LnJhbms7XG4gICAgICAgICAgICBmb3IgKGxldCBrID0gMDsgayA8IG1heE5EaW0gLSB4TkRpbTsgKytrKSB7XG4gICAgICAgICAgICAgIHggPSBLLmV4cGFuZERpbXMoeCwgMSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXNoYXBlZElucHV0cy5wdXNoKHgpO1xuICAgICAgICAgIH1cbiAgICAgICAgICByZXR1cm4gdGhpcy5tZXJnZUZ1bmN0aW9uKHJlc2hhcGVkSW5wdXRzKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAvLyBUcmFuc3Bvc2UgYWxsIGlucHV0cyBzbyB0aGF0IGJhdGNoIHNpemUgaXMgdGhlIGxhc3QgZGltZW5zaW9uLlxuICAgICAgICAgIC8vIFtiYXRjaFNpemUsIGRpbTEsIGRpbTIsIC4uLl0gLT4gW2RpbTEsIGRpbTIsIC4uLiwgYmF0Y2hTaXplXVxuICAgICAgICAgIGxldCB0cmFuc3Bvc2VkID0gZmFsc2U7XG4gICAgICAgICAgZm9yIChjb25zdCB4IG9mIGlucHV0cykge1xuICAgICAgICAgICAgY29uc3QgeE5EaW0gPSB4LnJhbms7XG4gICAgICAgICAgICBpZiAoeE5EaW0gPT0gbnVsbCkge1xuICAgICAgICAgICAgICBjb25zdCB4U2hhcGUgPSB4LnNoYXBlO1xuICAgICAgICAgICAgICBjb25zdCBiYXRjaFNpemUgPSB4U2hhcGVbMF07XG4gICAgICAgICAgICAgIGNvbnN0IG5ld1NoYXBlID0geFNoYXBlLnNsaWNlKDEpLmNvbmNhdChbYmF0Y2hTaXplXSk7XG4gICAgICAgICAgICAgIGxldCB4VHJhbnNwb3NlZCA9IHRmYy5yZXNoYXBlKFxuICAgICAgICAgICAgICAgICAgeCwgW2JhdGNoU2l6ZV0uY29uY2F0KG1hdGhVdGlscy5hcnJheVByb2QoeFNoYXBlLnNsaWNlKDEpKSkpO1xuICAgICAgICAgICAgICB4VHJhbnNwb3NlZCA9IHRmYy50cmFuc3Bvc2UoeFRyYW5zcG9zZWQsIFsxLCAwXSk7XG4gICAgICAgICAgICAgIHhUcmFuc3Bvc2VkID0gdGZjLnJlc2hhcGUoeFRyYW5zcG9zZWQsIG5ld1NoYXBlKTtcbiAgICAgICAgICAgICAgcmVzaGFwZWRJbnB1dHMucHVzaCh4VHJhbnNwb3NlZCk7XG4gICAgICAgICAgICAgIHRyYW5zcG9zZWQgPSB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIGlmICh4TkRpbSA+IDEpIHtcbiAgICAgICAgICAgICAgY29uc3QgZGltcyA9IG1hdGhVdGlscy5yYW5nZSgxLCB4TkRpbSkuY29uY2F0KFswXSk7XG4gICAgICAgICAgICAgIHJlc2hhcGVkSW5wdXRzLnB1c2godGZjLnRyYW5zcG9zZSh4LCBkaW1zKSk7XG4gICAgICAgICAgICAgIHRyYW5zcG9zZWQgPSB0cnVlO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgLy8gV2UgZG9uJ3QgdHJhbnNwb3NlIGlucHV0cyBpZiB0aGV5IGFyZSAxRCB2ZWN0b3JzIG9yIHNjYWxhcnMuXG4gICAgICAgICAgICAgIHJlc2hhcGVkSW5wdXRzLnB1c2goeCk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIGxldCB5ID0gdGhpcy5tZXJnZUZ1bmN0aW9uKHJlc2hhcGVkSW5wdXRzKTtcbiAgICAgICAgICBjb25zdCB5TkRpbSA9IHkucmFuaztcbiAgICAgICAgICBpZiAodHJhbnNwb3NlZCkge1xuICAgICAgICAgICAgLy8gSWYgaW5wdXRzIGhhdmUgYmVlbiB0cmFuc3Bvc2VkLCB3ZSBoYXZlIHRvIHRyYW5zcG9zZSB0aGUgb3V0cHV0XG4gICAgICAgICAgICAvLyB0b28uXG4gICAgICAgICAgICBpZiAoeU5EaW0gPT0gbnVsbCkge1xuICAgICAgICAgICAgICBjb25zdCB5U2hhcGUgPSB5LnNoYXBlO1xuICAgICAgICAgICAgICBjb25zdCB5TkRpbSA9IHlTaGFwZS5sZW5ndGg7XG4gICAgICAgICAgICAgIGNvbnN0IGJhdGNoU2l6ZSA9IHlTaGFwZVt5TkRpbSAtIDFdO1xuICAgICAgICAgICAgICBjb25zdCBuZXdTaGFwZSA9XG4gICAgICAgICAgICAgICAgICBbYmF0Y2hTaXplXS5jb25jYXQoeVNoYXBlLnNsaWNlKDAsIHlTaGFwZS5sZW5ndGggLSAxKSk7XG4gICAgICAgICAgICAgIHkgPSB0ZmMucmVzaGFwZShcbiAgICAgICAgICAgICAgICAgIHRmYy50cmFuc3Bvc2UodGZjLnJlc2hhcGUoeSwgWy0xLCBiYXRjaFNpemVdKSwgWzEsIDBdKSxcbiAgICAgICAgICAgICAgICAgIG5ld1NoYXBlKTtcbiAgICAgICAgICAgIH0gZWxzZSBpZiAoeU5EaW0gPiAxKSB7XG4gICAgICAgICAgICAgIGNvbnN0IGRpbXMgPSBbeU5EaW0gLSAxXS5jb25jYXQobWF0aFV0aWxzLnJhbmdlKDAsIHlORGltIC0gMSkpO1xuICAgICAgICAgICAgICB5ID0gdGZjLnRyYW5zcG9zZSh5LCBkaW1zKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgcmV0dXJuIHk7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiB0aGlzLm1lcmdlRnVuY3Rpb24oaW5wdXRzKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XG4gICAgaW5wdXRTaGFwZSA9IGlucHV0U2hhcGUgYXMgU2hhcGVbXTtcbiAgICBsZXQgb3V0cHV0U2hhcGU6IFNoYXBlO1xuICAgIGlmIChpbnB1dFNoYXBlWzBdID09IG51bGwpIHtcbiAgICAgIG91dHB1dFNoYXBlID0gbnVsbDtcbiAgICB9IGVsc2Uge1xuICAgICAgb3V0cHV0U2hhcGUgPSBpbnB1dFNoYXBlWzBdLnNsaWNlKDEpO1xuICAgIH1cbiAgICBmb3IgKGxldCBpID0gMTsgaSA8IGlucHV0U2hhcGUubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHNoYXBlID0gaW5wdXRTaGFwZVtpXSA9PSBudWxsID8gbnVsbCA6IGlucHV0U2hhcGVbaV0uc2xpY2UoMSk7XG4gICAgICBvdXRwdXRTaGFwZSA9IHRoaXMuY29tcHV0ZUVsZW1lbnR3aXNlT3BPdXRwdXRTaGFwZShvdXRwdXRTaGFwZSwgc2hhcGUpO1xuICAgIH1cblxuICAgIGxldCBiYXRjaFNpemVzOiBudW1iZXJbXSA9IFtdO1xuICAgIGZvciAoY29uc3Qgc2hhcGUgb2YgaW5wdXRTaGFwZSkge1xuICAgICAgaWYgKHNoYXBlICE9IG51bGwgJiYgc2hhcGVbMF0gIT09IG51bGwpIHtcbiAgICAgICAgYmF0Y2hTaXplcy5wdXNoKHNoYXBlWzBdKTtcbiAgICAgIH1cbiAgICB9XG4gICAgYmF0Y2hTaXplcyA9IGdlbmVyaWNfdXRpbHMudW5pcXVlKGJhdGNoU2l6ZXMpO1xuICAgIGlmIChiYXRjaFNpemVzLmxlbmd0aCA9PT0gMSkge1xuICAgICAgb3V0cHV0U2hhcGUgPSBiYXRjaFNpemVzLmNvbmNhdChvdXRwdXRTaGFwZSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIG91dHB1dFNoYXBlID0gW251bGxdLmNvbmNhdChvdXRwdXRTaGFwZSk7XG4gICAgfVxuICAgIHJldHVybiBvdXRwdXRTaGFwZTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXB1dGVNYXNrKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBtYXNrPzogVGVuc29yfFRlbnNvcltdKTpcbiAgICAgIFRlbnNvciB7XG4gICAgcmV0dXJuIHRmYy50aWR5KCgpID0+IHtcbiAgICAgIGlmIChtYXNrID09IG51bGwpIHtcbiAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICB9XG4gICAgICBpZiAoIUFycmF5LmlzQXJyYXkobWFzaykpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoJ2BtYXNrYCBzaG91bGQgYmUgYW4gQXJyYXknKTtcbiAgICAgIH1cbiAgICAgIGlmICghQXJyYXkuaXNBcnJheShpbnB1dHMpKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKCdgaW5wdXRzYCBzaG91bGQgYmUgYW4gQXJyYXknKTtcbiAgICAgIH1cbiAgICAgIGlmIChtYXNrLmxlbmd0aCAhPT0gaW5wdXRzLmxlbmd0aCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBUaGUgQXJyYXkgJ2lucHV0cycgYW5kICdtYXNrJyBhcmUgZXhwZWN0ZWQgdG8gaGF2ZSB0aGUgc2FtZSBgICtcbiAgICAgICAgICAgIGBsZW5ndGgsIGJ1dCBoYXZlIGRpZmZlcmVudCBsZW5ndGhzIGAgK1xuICAgICAgICAgICAgYCgke2lucHV0cy5sZW5ndGh9IHZzICR7bWFzay5sZW5ndGh9KWApO1xuICAgICAgfVxuICAgICAgaWYgKG1hc2suZXZlcnkobSA9PiBtID09IG51bGwpKSB7XG4gICAgICAgIHJldHVybiBudWxsO1xuICAgICAgfVxuICAgICAgbWFzayA9IG1hc2subWFwKG0gPT4gbSA9PSBudWxsID8gbSA6IHRmYy5leHBhbmREaW1zKG0sIDApKTtcbiAgICAgIGxldCBvdXRwdXQgPSBtYXNrWzBdO1xuICAgICAgZm9yIChsZXQgaSA9IDE7IGkgPCBtYXNrLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgICBvdXRwdXQgPSB0ZmMubG9naWNhbEFuZChvdXRwdXQsIG1hc2tbaV0pO1xuICAgICAgfVxuICAgICAgcmV0dXJuIG91dHB1dDtcbiAgICB9KTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgQWRkIGV4dGVuZHMgTWVyZ2Uge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdBZGQnO1xuICBjb25zdHJ1Y3RvcihhcmdzPzogTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgb3ZlcnJpZGUgbWVyZ2VGdW5jdGlvbihpbnB1dHM6IFRlbnNvcltdKTogVGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBsZXQgb3V0cHV0ID0gaW5wdXRzWzBdLmNsb25lKCk7XG4gICAgICBmb3IgKGxldCBpID0gMTsgaSA8IGlucHV0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICBvdXRwdXQgPSB0ZmMuYWRkKG91dHB1dCwgaW5wdXRzW2ldKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBvdXRwdXQ7XG4gICAgfSk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhBZGQpO1xuXG4vKipcbiAqIENhbGN1bGF0ZSB0aGUgZWxlbWVudC13aXNlIHN1bSBvZiBpbnB1dHMsIHdoaWNoIGFsbCBoYXZlIHRoZSBzYW1lIHNoYXBlLlxuICpcbiAqIFRoaXMgZnVuY3Rpb24gY2FuIGJlIGludm9rZWQgaW4gdGhyZWUgd2F5cy5cbiAqXG4gKiAxLiBDb25zdHJ1Y3QgYW4gaW5zdGFuY2Ugb2YgYEFkZGAgbGF5ZXIsIGJ5IHVzaW5nIG5vIGlucHV0IGFyZ3VtZW50XG4gKiAgICBvciBhIHNpbmdsZSBjb25maWd1cmF0aW9uIGFyZ3VtZW50LiBUaGUgcmVzdWx0YW50IGBBZGRgIGxheWVyIGNhbiB0aGVuXG4gKiAgICBiZSB1c2VkIG9uIGB0Zi5TeW1ib2xpY1RlbnNvcmBzIG9yIGB0Zi5UZW5zb3Jgcy4gRm9yIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IGFkZExheWVyID0gdGYubGF5ZXJzLmFkZCgpO1xuICpcbiAqIC8vIFRoZSBsYXllciBjYW4gYmUgYXBwbGllZCB0byBpbnB1dHMuXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IG91dHB1dCA9IGFkZExheWVyLmFwcGx5KFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2cob3V0cHV0LnNoYXBlKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDJdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLlxuICogYGBgXG4gKlxuICogMi4gSW52b2tlIGRpcmVjdGx5IG9uIGFuIGBBcnJheWAgb2YgYHRmLlN5bWJvbGljVGVuc29yYHMuIFRoaXMgY29uc3RydWN0c1xuICogICAgYW4gYExheWVyYCBvYmplY3QgaW50ZXJuYWxseSBhbmQgY2FsbHMgaXRzIGBhcHBseWAgbWV0aG9kIG9uIHRoZSBpbnB1dHMsXG4gKiAgICBnZW5lcmF0aW5nIGEgbmV3IGB0Zi5TeW1ib2xpY1RlbnNvcmAuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IG91dHB1dCA9IHRmLmxheWVycy5hZGQoW2lucHV0MSwgaW5wdXQyXSk7XG4gKiBjb25zb2xlLmxvZyhvdXRwdXQuc2hhcGUpO1xuICogLy8gWW91IGdldCBbbnVsbCwgMiwgMl0sIHdpdGggdGhlIGZpcnN0IGRpbWVuc2lvbiBhcyB0aGUgdW5kZXRlcm1pbmVkIGJhdGNoXG4gKiAvLyBkaW1lbnNpb24uXG4gKiBgYGBcbiAqXG4gKiAzLiBJbnZva2UgZGlyZWN0bHkgb24gYHRmLlRlbnNvcmBzLCBpLmUuLCBjb25jcmV0ZSB2YWx1ZXMuIFRoaXMgY29uc3RydWN0c1xuICogICAgYW4gYExheWVyYCBvYmplY3QgaW50ZXJuYWxseSBhbmQgY2FsbHMgaXRzIGBhcHBseWAgbWV0aG9kIG9uIHRoZSBpbnB1dHMsXG4gKiAgICBnZW5lcmF0aW5nIGEgbmV3IGB0Zi5UZW5zb3JgIGFzIHRoZSByZXN1bHQgb2YgdGhlIGNvbXB1dGF0aW9uLiBGb3JcbiAqIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IGlucHV0MSA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCA0XSwgWzIsIDJdKTtcbiAqIGNvbnN0IGlucHV0MiA9IHRmLnRlbnNvcjJkKFsxMCwgMjAsIDMwLCA0MF0sIFsyLCAyXSk7XG4gKiB0Zi5sYXllcnMuYWRkKFtpbnB1dDEsIGlucHV0Ml0pLnByaW50KCk7XG4gKiAvLyBHaXZlcyBbWzExLCAyMl0sIFszMywgNDRdXS5cbiAqXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBhZGQoY29uZmlnPzogU3ltYm9saWNUZW5zb3JbXXxUZW5zb3JbXXxMYXllckFyZ3MpOiBMYXllcnxcbiAgICBTeW1ib2xpY1RlbnNvcnxUZW5zb3Ige1xuICBpZiAoQXJyYXkuaXNBcnJheShjb25maWcpKSB7XG4gICAgY29uc3QgbGF5ZXIgPSBuZXcgQWRkKHt9KTtcbiAgICByZXR1cm4gbGF5ZXIuYXBwbHkoY29uZmlnKSBhcyBTeW1ib2xpY1RlbnNvciB8IFRlbnNvcjtcbiAgfSBlbHNlIHtcbiAgICByZXR1cm4gbmV3IEFkZChjb25maWcpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBNdWx0aXBseSBleHRlbmRzIE1lcmdlIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnTXVsdGlwbHknO1xuICBjb25zdHJ1Y3RvcihhcmdzPzogTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgb3ZlcnJpZGUgbWVyZ2VGdW5jdGlvbihpbnB1dHM6IFRlbnNvcltdKTogVGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBsZXQgb3V0cHV0ID0gaW5wdXRzWzBdLmNsb25lKCk7XG4gICAgICBmb3IgKGxldCBpID0gMTsgaSA8IGlucHV0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICBvdXRwdXQgPSB0ZmMubXVsKG91dHB1dCwgaW5wdXRzW2ldKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBvdXRwdXQ7XG4gICAgfSk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhNdWx0aXBseSk7XG5cbi8qKlxuICogQ2FsY3VsYXRlIHRoZSBlbGVtZW50LXdpc2UgcHJvZHVjdCBvZiBpbnB1dHMsIHdoaWNoIGFsbCBoYXZlIHRoZSBzYW1lIHNoYXBlLlxuICpcbiAqIFRoaXMgZnVuY3Rpb24gY2FuIGJlIGludm9rZWQgaW4gdGhyZWUgd2F5cy5cbiAqXG4gKiAxLiBDb25zdHJ1Y3QgYW4gaW5zdGFuY2Ugb2YgYE11bHRpcGx5YCBsYXllciwgYnkgdXNpbmcgbm8gaW5wdXQgYXJndW1lbnRcbiAqICAgIG9yIGEgc2luZ2xlIGNvbmZpZ3VyYXRpb24gYXJndW1lbnQuIFRoZSByZXN1bHRhbnQgYE11bHRpcGx5YCBsYXllciBjYW5cbiAqICAgIHRoZW4gYmUgdXNlZCBvbiBgdGYuU3ltYm9saWNUZW5zb3JgcyBvciBgdGYuVGVuc29yYHMuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBtdWx0aXBseUxheWVyID0gdGYubGF5ZXJzLm11bHRpcGx5KCk7XG4gKlxuICogLy8gVGhlIGxheWVyIGNhbiBiZSBhcHBsaWVkIHRvIGlucHV0cy5cbiAqIGNvbnN0IGlucHV0MSA9IHRmLmlucHV0KHtzaGFwZTogWzIsIDJdfSk7XG4gKiBjb25zdCBpbnB1dDIgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3Qgb3V0cHV0ID0gbXVsdGlwbHlMYXllci5hcHBseShbaW5wdXQxLCBpbnB1dDJdKTtcbiAqIGNvbnNvbGUubG9nKG91dHB1dC5zaGFwZSk7XG4gKiAvLyBZb3UgZ2V0IFtudWxsLCAyLCAyXSwgd2l0aCB0aGUgZmlyc3QgZGltZW5zaW9uIGFzIHRoZSB1bmRldGVybWluZWQgYmF0Y2hcbiAqIC8vIGRpbWVuc2lvbi5cbiAqIGBgYFxuICpcbiAqIDIuIEludm9rZSBkaXJlY3RseSBvbiBhbiBgQXJyYXlgIG9mIGB0Zi5TeW1ib2xpY1RlbnNvcmBzLiBUaGlzIGNvbnN0cnVjdHNcbiAqICAgIGFuIGBMYXllcmAgb2JqZWN0IGludGVybmFsbHkgYW5kIGNhbGxzIGl0cyBgYXBwbHlgIG1ldGhvZCBvbiB0aGUgaW5wdXRzLFxuICogICAgZ2VuZXJhdGluZyBhIG5ldyBgdGYuU3ltYm9saWNUZW5zb3JgLiBGb3IgZXhhbXBsZTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgaW5wdXQxID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IGlucHV0MiA9IHRmLmlucHV0KHtzaGFwZTogWzIsIDJdfSk7XG4gKiBjb25zdCBvdXRwdXQgPSB0Zi5sYXllcnMubXVsdGlwbHkoW2lucHV0MSwgaW5wdXQyXSk7XG4gKiBjb25zb2xlLmxvZyhvdXRwdXQuc2hhcGUpO1xuICogLy8gWW91IGdldCBbbnVsbCwgMiwgMl0sIHdpdGggdGhlIGZpcnN0IGRpbWVuc2lvbiBhcyB0aGUgdW5kZXRlcm1pbmVkIGJhdGNoXG4gKiAvLyBkaW1lbnNpb24uXG4gKiBgYGBcbiAqXG4gKiAzLiBJbnZva2UgZGlyZWN0bHkgb24gYHRmLlRlbnNvcmBzLCBpLmUuLCBjb25jcmV0ZSB2YWx1ZXMuIFRoaXMgY29uc3RydWN0c1xuICogICAgYW4gYExheWVyYCBvYmplY3QgaW50ZXJuYWxseSBhbmQgY2FsbHMgaXRzIGBhcHBseWAgbWV0aG9kIG9uIHRoZSBpbnB1dHMsXG4gKiAgICBnZW5lcmF0aW5nIGEgbmV3IGB0Zi5UZW5zb3JgIGFzIHRoZSByZXN1bHQgb2YgdGhlIGNvbXB1dGF0aW9uLiBGb3JcbiAqIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IGlucHV0MSA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCA0XSwgWzIsIDJdKTtcbiAqIGNvbnN0IGlucHV0MiA9IHRmLnRlbnNvcjJkKFsxMCwgMjAsIDMwLCA0MF0sIFsyLCAyXSk7XG4gKiB0Zi5sYXllcnMubXVsdGlwbHkoW2lucHV0MSwgaW5wdXQyXSkucHJpbnQoKTtcbiAqIC8vIEdpdmVzIFtbMTAsIDQwXSwgWzkwLCAxNjBdXS5cbiAqXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBtdWx0aXBseShjb25maWc/OiBTeW1ib2xpY1RlbnNvcltdfFRlbnNvcltdfExheWVyQXJncyk6IExheWVyfFxuICAgIFN5bWJvbGljVGVuc29yfFRlbnNvciB7XG4gIGlmIChBcnJheS5pc0FycmF5KGNvbmZpZykpIHtcbiAgICBjb25zdCBsYXllciA9IG5ldyBNdWx0aXBseSh7fSk7XG4gICAgcmV0dXJuIGxheWVyLmFwcGx5KGNvbmZpZykgYXMgU3ltYm9saWNUZW5zb3IgfCBUZW5zb3I7XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIG5ldyBNdWx0aXBseShjb25maWcpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBBdmVyYWdlIGV4dGVuZHMgTWVyZ2Uge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdBdmVyYWdlJztcbiAgY29uc3RydWN0b3IoYXJncz86IExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICB9XG5cbiAgcHJvdGVjdGVkIG92ZXJyaWRlIG1lcmdlRnVuY3Rpb24oaW5wdXRzOiBUZW5zb3JbXSk6IFRlbnNvciB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgbGV0IG91dHB1dCA9IGlucHV0c1swXS5jbG9uZSgpO1xuICAgICAgZm9yIChsZXQgaSA9IDE7IGkgPCBpbnB1dHMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgb3V0cHV0ID0gdGZjLmFkZChvdXRwdXQsIGlucHV0c1tpXSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gdGZjLm11bCgxIC8gaW5wdXRzLmxlbmd0aCwgb3V0cHV0KTtcbiAgICB9KTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEF2ZXJhZ2UpO1xuXG4vKipcbiAqIENhbGN1bGF0ZSB0aGUgZWxlbWVudC13aXNlIGFyaXRobWV0aWMgbWVhbiBvZiBpbnB1dHMsIHdoaWNoIGFsbCBoYXZlIHRoZSBzYW1lXG4gKiBzaGFwZS5cbiAqXG4gKiBUaGlzIGZ1bmN0aW9uIGNhbiBiZSBpbnZva2VkIGluIHRocmVlIHdheXMuXG4gKlxuICogMS4gQ29uc3RydWN0IGFuIGluc3RhbmNlIG9mIGBBdmVyYWdlYCBsYXllciwgYnkgdXNpbmcgbm8gaW5wdXQgYXJndW1lbnRcbiAqICAgIG9yIGEgc2luZ2xlIGNvbmZpZ3VyYXRpb24gYXJndW1lbnQuIFRoZSByZXN1bHRhbnQgYEF2ZXJhZ2VgIGxheWVyIGNhbiB0aGVuXG4gKiAgICBiZSB1c2VkIG9uIGB0Zi5TeW1ib2xpY1RlbnNvcmBzIG9yIGB0Zi5UZW5zb3Jgcy4gRm9yIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IGF2ZXJhZ2VMYXllciA9IHRmLmxheWVycy5hdmVyYWdlKCk7XG4gKlxuICogLy8gVGhlIGxheWVyIGNhbiBiZSBhcHBsaWVkIHRvIGlucHV0cy5cbiAqIGNvbnN0IGlucHV0MSA9IHRmLmlucHV0KHtzaGFwZTogWzIsIDJdfSk7XG4gKiBjb25zdCBpbnB1dDIgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3Qgb3V0cHV0ID0gYXZlcmFnZUxheWVyLmFwcGx5KFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2cob3V0cHV0LnNoYXBlKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDJdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLlxuICogYGBgXG4gKlxuICogMi4gSW52b2tlIGRpcmVjdGx5IG9uIGFuIGBBcnJheWAgb2YgYHRmLlN5bWJvbGljVGVuc29yYHMuIFRoaXMgY29uc3RydWN0c1xuICogICAgYW4gYExheWVyYCBvYmplY3QgaW50ZXJuYWxseSBhbmQgY2FsbHMgaXRzIGBhcHBseWAgbWV0aG9kIG9uIHRoZSBpbnB1dHMsXG4gKiAgICBnZW5lcmF0aW5nIGEgbmV3IGB0Zi5TeW1ib2xpY1RlbnNvcmAuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IG91dHB1dCA9IHRmLmxheWVycy5hdmVyYWdlKFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2cob3V0cHV0LnNoYXBlKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDJdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLlxuICogYGBgXG4gKlxuICogMy4gSW52b2tlIGRpcmVjdGx5IG9uIGB0Zi5UZW5zb3JgcywgaS5lLiwgY29uY3JldGUgdmFsdWVzLiBUaGlzIGNvbnN0cnVjdHNcbiAqICAgIGFuIGBMYXllcmAgb2JqZWN0IGludGVybmFsbHkgYW5kIGNhbGxzIGl0cyBgYXBwbHlgIG1ldGhvZCBvbiB0aGUgaW5wdXRzLFxuICogICAgZ2VuZXJhdGluZyBhIG5ldyBgdGYuVGVuc29yYCBhcyB0aGUgcmVzdWx0IG9mIHRoZSBjb21wdXRhdGlvbi4gRm9yXG4gKiBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgNF0sIFsyLCAyXSk7XG4gKiBjb25zdCBpbnB1dDIgPSB0Zi50ZW5zb3IyZChbMTAsIDIwLCAzMCwgNDBdLCBbMiwgMl0pO1xuICogdGYubGF5ZXJzLmF2ZXJhZ2UoW2lucHV0MSwgaW5wdXQyXSkucHJpbnQoKTtcbiAqIC8vIEdpdmVzIFtbNS41LCAxMV0sIFsxNi41LCAyMl1dLlxuICpcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGF2ZXJhZ2UoY29uZmlnPzogU3ltYm9saWNUZW5zb3JbXXxUZW5zb3JbXXxMYXllckFyZ3MpOiBMYXllcnxcbiAgICBTeW1ib2xpY1RlbnNvcnxUZW5zb3Ige1xuICBpZiAoQXJyYXkuaXNBcnJheShjb25maWcpKSB7XG4gICAgY29uc3QgbGF5ZXIgPSBuZXcgQXZlcmFnZSh7fSk7XG4gICAgcmV0dXJuIGxheWVyLmFwcGx5KGNvbmZpZykgYXMgU3ltYm9saWNUZW5zb3IgfCBUZW5zb3I7XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIG5ldyBBdmVyYWdlKGNvbmZpZyk7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIE1heGltdW0gZXh0ZW5kcyBNZXJnZSB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ01heGltdW0nO1xuICBjb25zdHJ1Y3RvcihhcmdzPzogTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgb3ZlcnJpZGUgbWVyZ2VGdW5jdGlvbihpbnB1dHM6IFRlbnNvcltdKTogVGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBsZXQgb3V0cHV0ID0gaW5wdXRzWzBdO1xuICAgICAgZm9yIChsZXQgaSA9IDE7IGkgPCBpbnB1dHMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgb3V0cHV0ID0gdGZjLm1heGltdW0ob3V0cHV0LCBpbnB1dHNbaV0pO1xuICAgICAgfVxuICAgICAgcmV0dXJuIG91dHB1dDtcbiAgICB9KTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKE1heGltdW0pO1xuXG4vKipcbiAqIENhbGN1bGF0ZSB0aGUgZWxlbWVudC13aXNlIG1heGltdW0gb2YgaW5wdXRzLCB3aGljaCBhbGwgaGF2ZSB0aGUgc2FtZSBzaGFwZS5cbiAqXG4gKiBUaGlzIGZ1bmN0aW9uIGNhbiBiZSBpbnZva2VkIGluIHRocmVlIHdheXMuXG4gKlxuICogMS4gQ29uc3RydWN0IGFuIGluc3RhbmNlIG9mIGBNYXhpbXVtYCBsYXllciwgYnkgdXNpbmcgbm8gaW5wdXQgYXJndW1lbnRcbiAqICAgIG9yIGEgc2luZ2xlIGNvbmZpZ3VyYXRpb24gYXJndW1lbnQuIFRoZSByZXN1bHRhbnQgYE1heGltdW1gIGxheWVyIGNhbiB0aGVuXG4gKiAgICBiZSB1c2VkIG9uIGB0Zi5TeW1ib2xpY1RlbnNvcmBzIG9yIGB0Zi5UZW5zb3Jgcy4gRm9yIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IG1heGltdW1MYXllciA9IHRmLmxheWVycy5tYXhpbXVtKCk7XG4gKlxuICogLy8gVGhlIGxheWVyIGNhbiBiZSBhcHBsaWVkIHRvIGlucHV0cy5cbiAqIGNvbnN0IGlucHV0MSA9IHRmLmlucHV0KHtzaGFwZTogWzIsIDJdfSk7XG4gKiBjb25zdCBpbnB1dDIgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3Qgb3V0cHV0ID0gbWF4aW11bUxheWVyLmFwcGx5KFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2cob3V0cHV0LnNoYXBlKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDJdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLlxuICogYGBgXG4gKlxuICogMi4gSW52b2tlIGRpcmVjdGx5IG9uIGFuIGBBcnJheWAgb2YgYHRmLlN5bWJvbGljVGVuc29yYHMuIFRoaXMgY29uc3RydWN0c1xuICogICAgYW4gYExheWVyYCBvYmplY3QgaW50ZXJuYWxseSBhbmQgY2FsbHMgaXRzIGBhcHBseWAgbWV0aG9kIG9uIHRoZSBpbnB1dHMsXG4gKiAgICBnZW5lcmF0aW5nIGEgbmV3IGB0Zi5TeW1ib2xpY1RlbnNvcmAuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IG91dHB1dCA9IHRmLmxheWVycy5tYXhpbXVtKFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2cob3V0cHV0LnNoYXBlKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDJdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLlxuICogYGBgXG4gKlxuICogMy4gSW52b2tlIGRpcmVjdGx5IG9uIGB0Zi5UZW5zb3JgcywgaS5lLiwgY29uY3JldGUgdmFsdWVzLiBUaGlzIGNvbnN0cnVjdHNcbiAqICAgIGFuIGBMYXllcmAgb2JqZWN0IGludGVybmFsbHkgYW5kIGNhbGxzIGl0cyBgYXBwbHlgIG1ldGhvZCBvbiB0aGUgaW5wdXRzLFxuICogICAgZ2VuZXJhdGluZyBhIG5ldyBgdGYuVGVuc29yYCBhcyB0aGUgcmVzdWx0IG9mIHRoZSBjb21wdXRhdGlvbi4gRm9yXG4gKiBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi50ZW5zb3IyZChbMSwgMjAsIDMsIDQwXSwgWzIsIDJdKTtcbiAqIGNvbnN0IGlucHV0MiA9IHRmLnRlbnNvcjJkKFsxMCwgMiwgMzAsIDRdLCBbMiwgMl0pO1xuICogdGYubGF5ZXJzLm1heGltdW0oW2lucHV0MSwgaW5wdXQyXSkucHJpbnQoKTtcbiAqIC8vIEdpdmVzIFtbMTAsIDIwXSwgWzMwLCA0MF1dLlxuICpcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG1heGltdW0oY29uZmlnPzogU3ltYm9saWNUZW5zb3JbXXxUZW5zb3JbXXxMYXllckFyZ3MpOiBMYXllcnxcbiAgICBTeW1ib2xpY1RlbnNvcnxUZW5zb3Ige1xuICBpZiAoQXJyYXkuaXNBcnJheShjb25maWcpKSB7XG4gICAgY29uc3QgbGF5ZXIgPSBuZXcgTWF4aW11bSh7fSk7XG4gICAgcmV0dXJuIGxheWVyLmFwcGx5KGNvbmZpZykgYXMgU3ltYm9saWNUZW5zb3IgfCBUZW5zb3I7XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIG5ldyBNYXhpbXVtKGNvbmZpZyk7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIE1pbmltdW0gZXh0ZW5kcyBNZXJnZSB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ01pbmltdW0nO1xuICBjb25zdHJ1Y3RvcihhcmdzPzogTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgb3ZlcnJpZGUgbWVyZ2VGdW5jdGlvbihpbnB1dHM6IFRlbnNvcltdKTogVGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBsZXQgb3V0cHV0ID0gaW5wdXRzWzBdO1xuICAgICAgZm9yIChsZXQgaSA9IDE7IGkgPCBpbnB1dHMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgb3V0cHV0ID0gdGZjLm1pbmltdW0ob3V0cHV0LCBpbnB1dHNbaV0pO1xuICAgICAgfVxuICAgICAgcmV0dXJuIG91dHB1dDtcbiAgICB9KTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKE1pbmltdW0pO1xuXG4vKipcbiAqIENhbGN1bGF0ZSB0aGUgZWxlbWVudC13aXNlIG1pbmltdW0gb2YgaW5wdXRzLCB3aGljaCBhbGwgaGF2ZSB0aGUgc2FtZSBzaGFwZS5cbiAqXG4gKiBUaGlzIGZ1bmN0aW9uIGNhbiBiZSBpbnZva2VkIGluIHRocmVlIHdheXMuXG4gKlxuICogMS4gQ29uc3RydWN0IGFuIGluc3RhbmNlIG9mIGBNaW5pbXVtYCBsYXllciwgYnkgdXNpbmcgbm8gaW5wdXQgYXJndW1lbnRcbiAqICAgIG9yIGEgc2luZ2xlIGNvbmZpZ3VyYXRpb24gYXJndW1lbnQuIFRoZSByZXN1bHRhbnQgYE1pbmltdW1gIGxheWVyIGNhbiB0aGVuXG4gKiAgICBiZSB1c2VkIG9uIGB0Zi5TeW1ib2xpY1RlbnNvcmBzIG9yIGB0Zi5UZW5zb3Jgcy4gRm9yIGV4YW1wbGU6XG4gKlxuICogYGBganNcbiAqIGNvbnN0IG1pbmltdW1MYXllciA9IHRmLmxheWVycy5taW5pbXVtKCk7XG4gKlxuICogLy8gVGhlIGxheWVyIGNhbiBiZSBhcHBsaWVkIHRvIGlucHV0cy5cbiAqIGNvbnN0IGlucHV0MSA9IHRmLmlucHV0KHtzaGFwZTogWzIsIDJdfSk7XG4gKiBjb25zdCBpbnB1dDIgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3Qgb3V0cHV0ID0gbWluaW11bUxheWVyLmFwcGx5KFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2cob3V0cHV0LnNoYXBlKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDJdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLlxuICogYGBgXG4gKlxuICogMi4gSW52b2tlIGRpcmVjdGx5IG9uIGFuIGBBcnJheWAgb2YgYHRmLlN5bWJvbGljVGVuc29yYHMuIFRoaXMgY29uc3RydWN0c1xuICogICAgYW4gYExheWVyYCBvYmplY3QgaW50ZXJuYWxseSBhbmQgY2FsbHMgaXRzIGBhcHBseWAgbWV0aG9kIG9uIHRoZSBpbnB1dHMsXG4gKiAgICBnZW5lcmF0aW5nIGEgbmV3IGB0Zi5TeW1ib2xpY1RlbnNvcmAuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAyXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAqIGNvbnN0IG91dHB1dCA9IHRmLmxheWVycy5taW5pbXVtKFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2cob3V0cHV0LnNoYXBlKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDJdLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uLlxuICogYGBgXG4gKlxuICogMy4gSW52b2tlIGRpcmVjdGx5IG9uIGB0Zi5UZW5zb3JgcywgaS5lLiwgY29uY3JldGUgdmFsdWVzLiBUaGlzIGNvbnN0cnVjdHNcbiAqICAgIGFuIGBMYXllcmAgb2JqZWN0IGludGVybmFsbHkgYW5kIGNhbGxzIGl0cyBgYXBwbHlgIG1ldGhvZCBvbiB0aGUgaW5wdXRzLFxuICogICAgZ2VuZXJhdGluZyBhIG5ldyBgdGYuVGVuc29yYCBhcyB0aGUgcmVzdWx0IG9mIHRoZSBjb21wdXRhdGlvbi4gRm9yXG4gKiBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi50ZW5zb3IyZChbMSwgMjAsIDMsIDQwXSwgWzIsIDJdKTtcbiAqIGNvbnN0IGlucHV0MiA9IHRmLnRlbnNvcjJkKFsxMCwgMiwgMzAsIDRdLCBbMiwgMl0pO1xuICogdGYubGF5ZXJzLm1pbmltdW0oW2lucHV0MSwgaW5wdXQyXSkucHJpbnQoKTtcbiAqIC8vIEdpdmVzIFtbMSwgMl0sIFszLCA0XV0uXG4gKlxuICovXG5leHBvcnQgZnVuY3Rpb24gbWluaW11bShjb25maWc/OiBTeW1ib2xpY1RlbnNvcltdfFRlbnNvcltdfExheWVyQXJncyk6IExheWVyfFxuICAgIFN5bWJvbGljVGVuc29yfFRlbnNvciB7XG4gIGlmIChBcnJheS5pc0FycmF5KGNvbmZpZykpIHtcbiAgICBjb25zdCBsYXllciA9IG5ldyBNaW5pbXVtKHt9KTtcbiAgICByZXR1cm4gbGF5ZXIuYXBwbHkoY29uZmlnKSBhcyBTeW1ib2xpY1RlbnNvciB8IFRlbnNvcjtcbiAgfSBlbHNlIHtcbiAgICByZXR1cm4gbmV3IE1pbmltdW0oY29uZmlnKTtcbiAgfVxufVxuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgQ29uY2F0ZW5hdGVMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogQXhpcyBhbG9uZyB3aGljaCB0byBjb25jYXRlbmF0ZS5cbiAgICovXG4gIGF4aXM/OiBudW1iZXI7XG59XG5cbmV4cG9ydCBjbGFzcyBDb25jYXRlbmF0ZSBleHRlbmRzIE1lcmdlIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnQ29uY2F0ZW5hdGUnO1xuICByZWFkb25seSBERUZBVUxUX0FYSVMgPSAtMTtcbiAgcHJpdmF0ZSByZWFkb25seSBheGlzOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoYXJncz86IENvbmNhdGVuYXRlTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gICAgaWYgKGFyZ3MgPT0gbnVsbCkge1xuICAgICAgYXJncyA9IHt9O1xuICAgIH1cbiAgICB0aGlzLmF4aXMgPSBhcmdzLmF4aXMgPT0gbnVsbCA/IHRoaXMuREVGQVVMVF9BWElTIDogYXJncy5heGlzO1xuICAgIHRoaXMuc3VwcG9ydHNNYXNraW5nID0gdHJ1ZTtcbiAgICB0aGlzLnJlc2hhcGVSZXF1aXJlZCA9IGZhbHNlO1xuICB9XG5cbiAgb3ZlcnJpZGUgYnVpbGQoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IHZvaWQge1xuICAgIC8vIFVzZWQgcHVyZWx5IGZvciBzaGFwZSB2YWxpZGF0aW9uLl1cbiAgICBpZiAoIShBcnJheS5pc0FycmF5KGlucHV0U2hhcGUpICYmIEFycmF5LmlzQXJyYXkoaW5wdXRTaGFwZVswXSkpIHx8XG4gICAgICAgIGlucHV0U2hhcGUubGVuZ3RoID09PSAxKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAnQSBgQ29uY2F0ZW5hdGVgIGxheWVyIHNob3VsZCBiZSBjYWxsZWQgb24gYSBsaXN0IG9mIGF0IGxlYXN0IDIgJyArXG4gICAgICAgICAgJ2lucHV0cycpO1xuICAgIH1cbiAgICBpbnB1dFNoYXBlID0gaW5wdXRTaGFwZSBhcyBTaGFwZVtdO1xuXG4gICAgbGV0IGFsbE5vbmVTaGFwZSA9IHRydWU7XG4gICAgZm9yIChjb25zdCBzaGFwZSBvZiBpbnB1dFNoYXBlKSB7XG4gICAgICBpZiAoc2hhcGUgIT0gbnVsbCkge1xuICAgICAgICBhbGxOb25lU2hhcGUgPSBmYWxzZTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgfVxuICAgIGlmIChhbGxOb25lU2hhcGUpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBjb25zdCBzaGFwZVNldDogU2hhcGVbXSA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgaW5wdXRTaGFwZS5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3Qgc2hhcGVXaXRob3V0Q29uY2F0QXhpcyA9IGlucHV0U2hhcGVbaV0uc2xpY2UoKTtcbiAgICAgIHNoYXBlV2l0aG91dENvbmNhdEF4aXMuc3BsaWNlKHRoaXMuYXhpcywgMSk7XG4gICAgICBsZXQgZXhpc3RzID0gZmFsc2U7XG4gICAgICBmb3IgKGNvbnN0IHNoYXBlIG9mIHNoYXBlU2V0KSB7XG4gICAgICAgIGlmICh1dGlsLmFycmF5c0VxdWFsKHNoYXBlLCBzaGFwZVdpdGhvdXRDb25jYXRBeGlzKSkge1xuICAgICAgICAgIGV4aXN0cyA9IHRydWU7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmICghZXhpc3RzKSB7XG4gICAgICAgIHNoYXBlU2V0LnB1c2goc2hhcGVXaXRob3V0Q29uY2F0QXhpcyk7XG4gICAgICB9XG4gICAgfVxuICAgIGlmIChzaGFwZVNldC5sZW5ndGggPiAxKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAnQSBgQ29uY2F0ZW5hdGVgIGxheWVyIHJlcXVpcmVzIGlucHV0cyB3aXRoIG1hdGNoaW5nIHNoYXBlcyAnICtcbiAgICAgICAgICAnZXhjZXB0IGZvciB0aGUgY29uY2F0IGF4aXMuIEdvdCBpbnB1dCBzaGFwZXM6ICcgK1xuICAgICAgICAgIEpTT04uc3RyaW5naWZ5KGlucHV0U2hhcGUpKTtcbiAgICB9XG4gIH1cblxuICBwcm90ZWN0ZWQgb3ZlcnJpZGUgbWVyZ2VGdW5jdGlvbihpbnB1dHM6IFRlbnNvcltdKTogVGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICByZXR1cm4gSy5jb25jYXRlbmF0ZShpbnB1dHMsIHRoaXMuYXhpcyk7XG4gICAgfSk7XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIGlmICghKEFycmF5LmlzQXJyYXkoaW5wdXRTaGFwZSkgJiYgQXJyYXkuaXNBcnJheShpbnB1dFNoYXBlWzBdKSkpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdBIGBDb25jYXRlbmF0ZWAgbGF5ZXIgc2hvdWxkIGJlIGNhbGxlZCBvbiBhIGxpc3Qgb2YgaW5wdXRzLicpO1xuICAgIH1cbiAgICBjb25zdCBpbnB1dFNoYXBlcyA9IGlucHV0U2hhcGUgYXMgU2hhcGVbXTtcbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGlucHV0U2hhcGVzWzBdLnNsaWNlKCk7XG4gICAgY29uc3QgYXhpcyA9IHRoaXMuYXhpcyA8IDAgPyBvdXRwdXRTaGFwZS5sZW5ndGggKyB0aGlzLmF4aXMgOiB0aGlzLmF4aXM7XG4gICAgLy8gUG9ydGluZyBOb3RlOiB0aGUgbGluZSBhYm92ZSBpcyBiZWNhdXNlIFR5cGVTY3JpcHQgZG9lc24ndCBzdXBwb3J0XG4gICAgLy8gICBuZWdhdGl2ZSBpbmRpY2VzLlxuICAgIGZvciAoY29uc3Qgc2hhcGUgb2YgaW5wdXRTaGFwZXMuc2xpY2UoMSkpIHtcbiAgICAgIGlmIChvdXRwdXRTaGFwZVtheGlzXSA9PSBudWxsIHx8IHNoYXBlW2F4aXNdID09IG51bGwpIHtcbiAgICAgICAgb3V0cHV0U2hhcGVbYXhpc10gPSBudWxsO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIG91dHB1dFNoYXBlW2F4aXNdICs9IHNoYXBlW2F4aXNdO1xuICAgIH1cbiAgICByZXR1cm4gb3V0cHV0U2hhcGU7XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlTWFzayhpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwgbWFzaz86IFRlbnNvcnxUZW5zb3JbXSk6XG4gICAgICBUZW5zb3Ige1xuICAgIGlmIChtYXNrID09IG51bGwpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICBpZiAoIUFycmF5LmlzQXJyYXkobWFzaykpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKCdgbWFza2Agc2hvdWxkIGJlIGFuIGFycmF5IGZvciBDb25jYXRlbmF0ZScpO1xuICAgIH1cbiAgICBpZiAoIUFycmF5LmlzQXJyYXkoaW5wdXRzKSkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoJ2BpbnB1dHNgIHNob3VsZCBiZSBhbiBhcnJheSBmb3IgQ29uY2F0ZW5hdGUnKTtcbiAgICB9XG4gICAgaWYgKG1hc2subGVuZ3RoICE9PSBpbnB1dHMubGVuZ3RoKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgTWlzbWF0Y2ggaW4gdGhlIGxlbmd0aCBvZiBtYXNrICgke21hc2subGVuZ3RofSkgYCArXG4gICAgICAgICAgYGFuZCB0aGUgbGVnbnRoIG9mIGlucHV0cyAoJHtpbnB1dHMubGVuZ3RofSlgKTtcbiAgICB9XG4gICAgcmV0dXJuIHRmYy50aWR5KCgpID0+IHtcbiAgICAgIGxldCBhbGxOdWxsTWFza3MgPSB0cnVlO1xuICAgICAgbWFzay5mb3JFYWNoKG0gPT4ge1xuICAgICAgICBpZiAobSAhPSBudWxsKSB7XG4gICAgICAgICAgYWxsTnVsbE1hc2tzID0gZmFsc2U7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICAgIGlmIChhbGxOdWxsTWFza3MpIHtcbiAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICB9XG4gICAgICBjb25zdCBvdXRwdXRNYXNrczogVGVuc29yW10gPSBbXTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgaW5wdXRzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgIGlmIChtYXNrW2ldID09IG51bGwpIHtcbiAgICAgICAgICAvLyBJbnB1dCBpcyB1bm1hc2tlZC4gQXBwZW5kIGFsbCAxJ3MgdG8gbWFza3MuXG4gICAgICAgICAgb3V0cHV0TWFza3MucHVzaCh0ZmMuY2FzdCh0ZmMub25lc0xpa2UoaW5wdXRzW2ldKSwgJ2Jvb2wnKSk7XG4gICAgICAgIH0gZWxzZSBpZiAobWFza1tpXS5yYW5rIDwgaW5wdXRzW2ldLnJhbmspIHtcbiAgICAgICAgICAvLyBNYXNrIGlzIHNtYWxsZXIgdGhhbiB0aGUgaW5wdXQsIGV4cGFuZCBpdC5cbiAgICAgICAgICBvdXRwdXRNYXNrcy5wdXNoKHRmYy5leHBhbmREaW1zKG1hc2tbaV0sIC0xKSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgb3V0cHV0TWFza3MucHVzaChtYXNrW2ldKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgY29uc3QgY29uY2F0ZW5hdGVkTWFza3MgPSB0ZmMuY29uY2F0KG91dHB1dE1hc2tzLCB0aGlzLmF4aXMpO1xuICAgICAgcmV0dXJuIHRmYy5hbGwoY29uY2F0ZW5hdGVkTWFza3MsIC0xLCBmYWxzZSk7XG4gICAgfSk7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtcbiAgICAgICdheGlzJzogdGhpcy5heGlzLFxuICAgIH07XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQ29uY2F0ZW5hdGUpO1xuXG4vKipcbiAqIENvbmNhdGVuYXRlIGFuIGBBcnJheWAgb2YgaW5wdXRzLlxuICpcbiAqIFRoaXMgZnVuY3Rpb24gY2FuIGJlIGludm9rZWQgaW4gdGhyZWUgd2F5cy5cbiAqXG4gKiAxLiBDb25zdHJ1Y3QgYW4gaW5zdGFuY2Ugb2YgYENvbmNhdGVuYXRlYCBsYXllciwgYnkgdXNpbmcgbm8gaW5wdXQgYXJndW1lbnRcbiAqICAgIG9yIGEgc2luZ2xlIGNvbmZpZ3VyYXRpb24gYXJndW1lbnQuIFRoZSByZXN1bHRhbnQgYENvbmNhdGVuYXRlYCBsYXllciBjYW5cbiAqICAgIHRoZW4gYmUgdXNlZCBvbiBgdGYuU3ltYm9saWNUZW5zb3JgcyBvciBgdGYuVGVuc29yYHMuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBjb25jYXRMYXllciA9IHRmLmxheWVycy5jb25jYXRlbmF0ZSgpO1xuICpcbiAqIC8vIFRoZSBsYXllciBjYW4gYmUgYXBwbGllZCB0byBpbnB1dHMuXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAzXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgNF19KTtcbiAqIGNvbnN0IG91dHB1dCA9IGNvbmNhdExheWVyLmFwcGx5KFtpbnB1dDEsIGlucHV0Ml0pO1xuICogY29uc29sZS5sb2cob3V0cHV0LnNoYXBlKTtcbiAqIC8vIFlvdSBnZXQgW251bGwsIDIsIDddLCB3aXRoIHRoZSBmaXJzdCBkaW1lbnNpb24gYXMgdGhlIHVuZGV0ZXJtaW5lZCBiYXRjaFxuICogLy8gZGltZW5zaW9uIGFuZCB0aGUgbGFzdCBkaW1lbnNpb24gYXMgdGhlIHJlc3VsdCBvZiBjb25jYXRlbmF0aW5nIHRoZVxuICogLy8gbGFzdCBkaW1lbnNpb25zIG9mIHRoZSB0d28gaW5wdXRzLlxuICogYGBgXG4gKlxuICogMi4gSW52b2tlIGRpcmVjdGx5IG9uIGFuIGBBcnJheWAgb2YgYHRmLlN5bWJvbGljVGVuc29yYHMuIFRoaXMgY29uc3RydWN0c1xuICogICAgYW4gYExheWVyYCBvYmplY3QgaW50ZXJuYWxseSBhbmQgY2FsbHMgaXRzIGBhcHBseWAgbWV0aG9kIG9uIHRoZSBpbnB1dHMsXG4gKiAgICBnZW5lcmF0aW5nIGEgbmV3IGB0Zi5TeW1ib2xpY1RlbnNvcmAuIEZvciBleGFtcGxlOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbnB1dDEgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyLCAzXX0pO1xuICogY29uc3QgaW5wdXQyID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgNF19KTtcbiAqIGNvbnN0IG91dHB1dCA9IHRmLmxheWVycy5jb25jYXRlbmF0ZShbaW5wdXQxLCBpbnB1dDJdKTtcbiAqIGNvbnNvbGUubG9nKG91dHB1dC5zaGFwZSk7XG4gKiAvLyBZb3UgZ2V0IFtudWxsLCAyLCAyXSwgd2l0aCB0aGUgZmlyc3QgZGltZW5zaW9uIGFzIHRoZSB1bmRldGVybWluZWQgYmF0Y2hcbiAqIC8vIGRpbWVuc2lvbiBhbmQgdGhlIGxhc3QgZGltZW5zaW9uIGFzIHRoZSByZXN1bHQgb2YgY29uY2F0ZW5hdGluZyB0aGVcbiAqIC8vIGxhc3QgZGltZW5zaW9ucyBvZiB0aGUgdHdvIGlucHV0cy5cbiAqIGBgYFxuICpcbiAqIDMuIEludm9rZSBkaXJlY3RseSBvbiBgdGYuVGVuc29yYHMsIGkuZS4sIGNvbmNyZXRlIHZhbHVlcy4gVGhpcyBjb25zdHJ1Y3RzXG4gKiAgICBhbiBgTGF5ZXJgIG9iamVjdCBpbnRlcm5hbGx5IGFuZCBjYWxscyBpdHMgYGFwcGx5YCBtZXRob2Qgb24gdGhlIGlucHV0cyxcbiAqICAgIGdlbmVyYXRpbmcgYSBuZXcgYHRmLlRlbnNvcmAgYXMgdGhlIHJlc3VsdCBvZiB0aGUgY29tcHV0YXRpb24uIEZvclxuICogZXhhbXBsZTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgaW5wdXQxID0gdGYudGVuc29yMmQoW1sxLCAyXSwgWzMsIDRdXSwgWzIsIDJdKTtcbiAqIGNvbnN0IGlucHV0MiA9IHRmLnRlbnNvcjJkKFtbMTAsIDIwXSwgWzMwLCA0MF1dLCBbMiwgMl0pO1xuICogdGYubGF5ZXJzLmNvbmNhdGVuYXRlKFtpbnB1dDEsIGlucHV0Ml0pLnByaW50KCk7XG4gKiAvLyBHaXZlcyBbWzEsIDIsIDEwLCAyMF0sIFszLCA0LCAzMCwgNDBdXS5cbiAqXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjb25jYXRlbmF0ZShjb25maWc/OiBTeW1ib2xpY1RlbnNvcltdfFRlbnNvcltdfFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIENvbmNhdGVuYXRlTGF5ZXJBcmdzKTogTGF5ZXJ8U3ltYm9saWNUZW5zb3J8VGVuc29yIHtcbiAgaWYgKEFycmF5LmlzQXJyYXkoY29uZmlnKSkge1xuICAgIGNvbnN0IGxheWVyID0gbmV3IENvbmNhdGVuYXRlKHt9KTtcbiAgICByZXR1cm4gbGF5ZXIuYXBwbHkoY29uZmlnKSBhcyBTeW1ib2xpY1RlbnNvciB8IFRlbnNvcjtcbiAgfSBlbHNlIHtcbiAgICByZXR1cm4gbmV3IENvbmNhdGVuYXRlKGNvbmZpZyk7XG4gIH1cbn1cblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIERvdExheWVyQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIC8qKlxuICAgKiBBeGlzIG9yIGF4ZXMgYWxvbmcgd2hpY2ggdGhlIGRvdCBwcm9kdWN0IHdpbGwgYmUgdGFrZW4uXG4gICAqXG4gICAqIEludGVnZXIgb3IgYW4gQXJyYXkgb2YgaW50ZWdlcnMuXG4gICAqL1xuICBheGVzOiBudW1iZXJ8W251bWJlciwgbnVtYmVyXTtcblxuICAvKipcbiAgICogV2hldGhlciB0byBMMi1ub3JtYWxpemUgc2FtcGxlcyBhbG9uZyB0aGUgZG90IHByb2R1Y3QgYXhpc1xuICAgKiBiZWZvcmUgdGFraW5nIHRoZSBkb3QgcHJvZHVjdC5cbiAgICpcbiAgICogSWYgc2V0IHRvIGB0cnVlYCwgdGhlIG91dHB1dCBvZiB0aGUgZG90IHByb2R1Y3QgaXMgdGhlIGNvc2luZVxuICAgKiBwcm94aW1pdHkgYmV0d2VlbiB0aGUgdHdvIHNhbXBsZXMuXG4gICAqL1xuICBub3JtYWxpemU/OiBib29sZWFuO1xufVxuXG4vKipcbiAqIEludGVycHJldGFibGUgcG90ZW50aWFsbHkgbmVnYXRpdmUgYXhpcyBpbmRleC5cbiAqXG4gKiBGb3IgZXhhbXBsZSwgZ2l2ZW4gYXhpcyA9IC0xLCBhbmQgZGltID0gMywgdGhpcyBmdW5jdGlvbiB3aWxsIHJldHVybiAyLlxuICpcbiAqIEBwYXJhbSBheGlzIFRoZSBheGlzIGluZGV4LCBtYXkgYmUgYSBwb3NpdGl2ZSwgemVybyBvciBuZWdhdGl2ZSBpbnRlZ2VyLlxuICogQHBhcmFtIGRpbSBUb3RhbCBudW1iZXIgb2YgZGltZW5zaW9ucywgYSBwb3NpdGl2ZSBpbnRlZ2VyLlxuICogQHJldHVybnMgQSBub24tbmVnYXRpdmUgYXhpcyBpbmRleCBlcXVpdmFsZW50IHRvIHRoZSBpbnB1dCBgYXhpc2AuXG4gKi9cbmZ1bmN0aW9uIGludGVycHJldEF4aXMoYXhpczogbnVtYmVyLCBkaW06IG51bWJlcik6IG51bWJlciB7XG4gIHdoaWxlIChheGlzIDwgMCkge1xuICAgIGF4aXMgKz0gZGltO1xuICB9XG4gIHJldHVybiBheGlzO1xufVxuXG5mdW5jdGlvbiBiYXRjaERvdCh4OiBUZW5zb3IsIHk6IFRlbnNvciwgYXhlczogbnVtYmVyfFtudW1iZXIsIG51bWJlcl0pOiBUZW5zb3Ige1xuICBpZiAoeC5zaGFwZS5sZW5ndGggPiAzIHx8IHkuc2hhcGUubGVuZ3RoID4gMykge1xuICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAnYmF0Y2hEb3QgaXMgbm90IGltcGxlbWVudGVkIGZvciB0ZW5zb3JzIG9mIDREIG9yIGhpZ2hlciByYW5rIHlldCcpO1xuICB9XG4gIHRmYy51dGlsLmFzc2VydChcbiAgICAgIHguc2hhcGUubGVuZ3RoID49IDIsXG4gICAgICAoKSA9PiBgYmF0Y2hEb3QgcmVxdWlyZXMgdGhlIHJhbmsgb2YgeCB0byBiZSA+PSAyLCBgICtcbiAgICAgICAgICBgYnV0IGdvdCAke3guc2hhcGUubGVuZ3RofWApO1xuICB0ZmMudXRpbC5hc3NlcnQoXG4gICAgICB4LnNoYXBlLmxlbmd0aCA+PSAyLFxuICAgICAgKCkgPT4gYGJhdGNoRG90IHJlcXVpcmVzIHRoZSByYW5rIG9mIHkgdG8gYmUgPj0gMiwgYCArXG4gICAgICAgICAgYGJ1dCBnb3QgJHt5LnNoYXBlLmxlbmd0aH1gKTtcblxuICBpZiAodHlwZW9mIGF4ZXMgPT09ICdudW1iZXInKSB7XG4gICAgYXhlcyA9IFtheGVzLCBheGVzXTtcbiAgfVxuXG4gIGlmICh4LmR0eXBlID09PSAnY29tcGxleDY0JyB8fCB5LmR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAnYmF0Y2hEb3QgaXMgbm90IGltcGxlbWVudGVkIGZvciBjb21wbGV4NjQtdHlwZSBUZW5zb3JzIHlldC4nKTtcbiAgfVxuXG4gIGNvbnN0IHhORGltID0geC5zaGFwZS5sZW5ndGg7XG4gIGNvbnN0IHlORGltID0geS5zaGFwZS5sZW5ndGg7XG4gIGlmIChheGVzID09IG51bGwpIHtcbiAgICAvLyBCZWhhdmUgbGlrZSBiYXRjaE1hdG11bCBieSBkZWZhdWx0LlxuICAgIGF4ZXMgPSBbeE5EaW0gLSAxLCB5TkRpbSAtIDJdO1xuICB9XG4gIGNvbnN0IGF4ZXNBcnJheSA9IGF4ZXMgYXMgW251bWJlciwgbnVtYmVyXTtcblxuICByZXR1cm4gdGZjLnRpZHkoKCkgPT4ge1xuICAgIGxldCBkaWZmOiBudW1iZXI7XG4gICAgaWYgKHhORGltID4geU5EaW0pIHtcbiAgICAgIGRpZmYgPSB4TkRpbSAtIHlORGltO1xuICAgICAgY29uc3QgZGlmZlNoYXBlOiBTaGFwZSA9IFtdO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBkaWZmOyArK2kpIHtcbiAgICAgICAgZGlmZlNoYXBlLnB1c2goMSk7XG4gICAgICB9XG4gICAgICB5ID0gdGZjLnJlc2hhcGUoeSwgeS5zaGFwZS5jb25jYXQoZGlmZlNoYXBlKSk7XG4gICAgfSBlbHNlIGlmICh5TkRpbSA+IHhORGltKSB7XG4gICAgICBkaWZmID0geU5EaW0gLSB4TkRpbTtcbiAgICAgIGNvbnN0IGRpZmZTaGFwZTogU2hhcGUgPSBbXTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgZGlmZjsgKytpKSB7XG4gICAgICAgIGRpZmZTaGFwZS5wdXNoKDEpO1xuICAgICAgfVxuICAgICAgeCA9IHRmYy5yZXNoYXBlKHgsIHguc2hhcGUuY29uY2F0KGRpZmZTaGFwZSkpO1xuICAgIH0gZWxzZSB7XG4gICAgICBkaWZmID0gMDtcbiAgICB9XG5cbiAgICBsZXQgb3V0OiBUZW5zb3I7XG4gICAgaWYgKHguc2hhcGUubGVuZ3RoID09PSAyICYmIHkuc2hhcGUubGVuZ3RoID09PSAyKSB7XG4gICAgICBpZiAoYXhlc0FycmF5WzBdID09PSBheGVzQXJyYXlbMV0pIHtcbiAgICAgICAgb3V0ID0gdGZjLnN1bSh0ZmMubXVsKHgsIHkpLCBheGVzQXJyYXlbMF0pO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgb3V0ID0gdGZjLnN1bSh0ZmMubXVsKHRmYy50cmFuc3Bvc2UoeCwgWzEsIDBdKSwgeSksIGF4ZXNBcnJheVsxXSk7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGFkalggPSBheGVzQXJyYXlbMF0gIT09IHguc2hhcGUubGVuZ3RoIC0gMTtcbiAgICAgIGNvbnN0IGFkalkgPSBheGVzQXJyYXlbMV0gPT09IHkuc2hhcGUubGVuZ3RoIC0gMTtcbiAgICAgIG91dCA9IHRmYy5tYXRNdWwoeCwgeSwgYWRqWCwgYWRqWSk7XG4gICAgfVxuXG4gICAgaWYgKGRpZmYgPiAwKSB7XG4gICAgICBsZXQgaWR4OiBudW1iZXI7XG4gICAgICBpZiAoeE5EaW0gPiB5TkRpbSkge1xuICAgICAgICBpZHggPSB4TkRpbSArIHlORGltIC0gMztcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGlkeCA9IHhORGltIC0gMTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHNxdWVlemVBeGVzOiBudW1iZXJbXSA9IFtdO1xuICAgICAgZm9yIChsZXQgaSA9IGlkeDsgaSA8IGlkeCArIGRpZmY7ICsraSkge1xuICAgICAgICBzcXVlZXplQXhlcy5wdXNoKGkpO1xuICAgICAgfVxuICAgICAgb3V0ID0gdGZjLnNxdWVlemUob3V0LCBzcXVlZXplQXhlcyk7XG4gICAgfVxuICAgIGlmIChvdXQuc2hhcGUubGVuZ3RoID09PSAxKSB7XG4gICAgICBvdXQgPSB0ZmMuZXhwYW5kRGltcyhvdXQsIDEpO1xuICAgIH1cbiAgICByZXR1cm4gb3V0O1xuICB9KTtcbn1cblxuZXhwb3J0IGNsYXNzIERvdCBleHRlbmRzIE1lcmdlIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnRG90JztcblxuICBwcml2YXRlIGF4ZXM6IG51bWJlcnxbbnVtYmVyLCBudW1iZXJdO1xuICBwcml2YXRlIG5vcm1hbGl6ZTogYm9vbGVhbjtcblxuICBjb25zdHJ1Y3RvcihhcmdzOiBEb3RMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzKTtcbiAgICB0aGlzLmF4ZXMgPSBhcmdzLmF4ZXM7XG4gICAgdGhpcy5ub3JtYWxpemUgPSBhcmdzLm5vcm1hbGl6ZSA9PSBudWxsID8gZmFsc2UgOiBhcmdzLm5vcm1hbGl6ZTtcbiAgICB0aGlzLnN1cHBvcnRzTWFza2luZyA9IHRydWU7XG4gICAgdGhpcy5yZXNoYXBlUmVxdWlyZWQgPSBmYWxzZTtcbiAgfVxuXG4gIG92ZXJyaWRlIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICB0ZmMudXRpbC5hc3NlcnQoXG4gICAgICAgIEFycmF5LmlzQXJyYXkoaW5wdXRTaGFwZSkgJiYgaW5wdXRTaGFwZS5sZW5ndGggPT09IDIgJiZcbiAgICAgICAgICAgIEFycmF5LmlzQXJyYXkoaW5wdXRTaGFwZVswXSkgJiYgQXJyYXkuaXNBcnJheShpbnB1dFNoYXBlWzFdKSxcbiAgICAgICAgKCkgPT4gJ0EgYERvdGAgbGF5ZXIgc2hvdWxkIGJlIGNhbGxlZCBvbiBhIGxpc3Qgb2YgZXhhY3RseSAyIGlucHV0cy4nKTtcbiAgICBjb25zdCBzaGFwZTEgPSBpbnB1dFNoYXBlWzBdIGFzIFNoYXBlO1xuICAgIGNvbnN0IHNoYXBlMiA9IGlucHV0U2hhcGVbMV0gYXMgU2hhcGU7XG4gICAgaWYgKHNoYXBlMS5sZW5ndGggPiAzIHx8IHNoYXBlMi5sZW5ndGggPiAzKSB7XG4gICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAnRG90IGxheWVyIGRvZXMgbm90IHN1cHBvcnQgdGVuc29ycyBvZiA0RCBvciBoaWdoZXIgcmFuayB5ZXQuJyk7XG4gICAgfVxuXG4gICAgY29uc3QgYXhlcyA9IHRoaXMuaW50ZXJwcmV0QXhlcyhzaGFwZTEsIHNoYXBlMik7XG4gICAgaWYgKHNoYXBlMVtheGVzWzBdXSAhPT0gc2hhcGUyW2F4ZXNbMV1dKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgRGltZW5zaW9uIGluY29tcGF0aWJpbGl0eTogYCArXG4gICAgICAgICAgYCR7c2hhcGUxW2F4ZXNbMF1dfSAhPT0gJHtzaGFwZTJbYXhlc1sxXV19YCk7XG4gICAgfVxuICB9XG5cbiAgcHJvdGVjdGVkIG92ZXJyaWRlIG1lcmdlRnVuY3Rpb24oaW5wdXRzOiBUZW5zb3JbXSk6IFRlbnNvciB7XG4gICAgaWYgKGlucHV0cy5sZW5ndGggIT09IDIpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdBIGBEb3RgIGxheWVyIG11c3QgYmUgY2FsbGVkIG9uIGV4YWN0bHkgMiBpbnB1dHMsICcgK1xuICAgICAgICAgIGBidXQgcmVjZWl2ZWQgJHtpbnB1dHMubGVuZ3RofSBpbnB1dChzKS5gKTtcbiAgICB9XG5cbiAgICBsZXQgeDEgPSBpbnB1dHNbMF07XG4gICAgbGV0IHgyID0gaW5wdXRzWzFdO1xuICAgIGxldCBheGVzOiBbbnVtYmVyLCBudW1iZXJdO1xuICAgIGlmICghQXJyYXkuaXNBcnJheSh0aGlzLmF4ZXMpKSB7XG4gICAgICBheGVzID0gW1xuICAgICAgICBpbnRlcnByZXRBeGlzKHRoaXMuYXhlcywgeDEuc2hhcGUubGVuZ3RoKSxcbiAgICAgICAgaW50ZXJwcmV0QXhpcyh0aGlzLmF4ZXMsIHgyLnNoYXBlLmxlbmd0aClcbiAgICAgIF07XG4gICAgfSBlbHNlIHtcbiAgICAgIGF4ZXMgPSB0aGlzLmF4ZXMubWFwKFxuICAgICAgICAgICAgICAgICAoYXhpcywgaSkgPT4gaW50ZXJwcmV0QXhpcyhcbiAgICAgICAgICAgICAgICAgICAgIGF4aXMsIGlucHV0c1tpXS5zaGFwZS5sZW5ndGgpKSBhcyBbbnVtYmVyLCBudW1iZXJdO1xuICAgIH1cbiAgICBpZiAodGhpcy5ub3JtYWxpemUpIHtcbiAgICAgIHgxID0gbDJOb3JtYWxpemUoeDEsIGF4ZXNbMF0pO1xuICAgICAgeDIgPSBsMk5vcm1hbGl6ZSh4MiwgYXhlc1sxXSk7XG4gICAgfVxuICAgIHJldHVybiBiYXRjaERvdCh4MSwgeDIsIGF4ZXMpO1xuICB9XG5cbiAgcHJpdmF0ZSBpbnRlcnByZXRBeGVzKHNoYXBlMTogU2hhcGUsIHNoYXBlMjogU2hhcGUpOiBudW1iZXJbXSB7XG4gICAgbGV0IGF4ZXM6IG51bWJlcltdO1xuICAgIGlmICghQXJyYXkuaXNBcnJheSh0aGlzLmF4ZXMpKSB7XG4gICAgICAvLyBgdGhpcy5heGVzYCBpcyBhIHNpbmdsZSBpbnRlZ2VyLlxuICAgICAgYXhlcyA9IFtcbiAgICAgICAgaW50ZXJwcmV0QXhpcyh0aGlzLmF4ZXMsIHNoYXBlMS5sZW5ndGgpLFxuICAgICAgICBpbnRlcnByZXRBeGlzKHRoaXMuYXhlcywgc2hhcGUyLmxlbmd0aClcbiAgICAgIF07XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIGB0aGlzLmF4ZXNgIGlzIGFuIEFycmF5IG9mIGludGVnZXJzLlxuICAgICAgYXhlcyA9IHRoaXMuYXhlcztcbiAgICB9XG4gICAgcmV0dXJuIGF4ZXM7XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIHRmYy51dGlsLmFzc2VydChcbiAgICAgICAgQXJyYXkuaXNBcnJheShpbnB1dFNoYXBlKSAmJiBpbnB1dFNoYXBlLmxlbmd0aCA9PT0gMiAmJlxuICAgICAgICAgICAgQXJyYXkuaXNBcnJheShpbnB1dFNoYXBlWzBdKSAmJiBBcnJheS5pc0FycmF5KGlucHV0U2hhcGVbMV0pLFxuICAgICAgICAoKSA9PiAnQSBgRG90YCBsYXllciBzaG91bGQgYmUgY2FsbGVkIG9uIGEgbGlzdCBvZiBleGFjdGx5IDIgaW5wdXRzLicpO1xuICAgIGNvbnN0IHNoYXBlMSA9IChpbnB1dFNoYXBlWzBdIGFzIFNoYXBlKS5zbGljZSgpO1xuICAgIGNvbnN0IHNoYXBlMiA9IChpbnB1dFNoYXBlWzFdIGFzIFNoYXBlKS5zbGljZSgpO1xuICAgIGlmIChzaGFwZTEubGVuZ3RoID4gMyB8fCBzaGFwZTIubGVuZ3RoID4gMykge1xuICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgJ0RvdCBsYXllciBkb2VzIG5vdCBzdXBwb3J0IHRlbnNvcnMgb2YgNEQgb3IgaGlnaGVyIHJhbmsgeWV0LicpO1xuICAgIH1cblxuICAgIGNvbnN0IGF4ZXMgPSB0aGlzLmludGVycHJldEF4ZXMoc2hhcGUxLCBzaGFwZTIpO1xuICAgIHNoYXBlMS5zcGxpY2UoYXhlc1swXSwgMSk7XG4gICAgc2hhcGUyLnNwbGljZShheGVzWzFdLCAxKTtcbiAgICBzaGFwZTIuc3BsaWNlKDAsIDEpO1xuICAgIGNvbnN0IG91dHB1dFNoYXBlID0gc2hhcGUxLmNvbmNhdChzaGFwZTIpO1xuICAgIGlmIChvdXRwdXRTaGFwZS5sZW5ndGggPT09IDEpIHtcbiAgICAgIG91dHB1dFNoYXBlLnB1c2goMSk7XG4gICAgfVxuICAgIHJldHVybiBvdXRwdXRTaGFwZTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXB1dGVNYXNrKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBtYXNrPzogVGVuc29yfFRlbnNvcltdKTpcbiAgICAgIFRlbnNvciB7XG4gICAgcmV0dXJuIG51bGw7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtcbiAgICAgICdheGVzJzogdGhpcy5heGVzLFxuICAgICAgJ25vcm1hbGl6ZSc6IHRoaXMubm9ybWFsaXplXG4gICAgfTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhEb3QpO1xuXG4vLyBUT0RPKGNhaXMpOiBBZGQgZnVuY3Rpb25hbCBpbnRlcmZhY2VzIGZvciB0aGUgbWVyZ2UgbGF5ZXJzLlxuIl19