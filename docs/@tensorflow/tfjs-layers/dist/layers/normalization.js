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
 * Normalization layers.
 */
import * as tfc from '@tensorflow/tfjs-core';
import { moments, reshape, serialization, tidy, util } from '@tensorflow/tfjs-core';
import { getConstraint, serializeConstraint } from '../constraints';
import { InputSpec, Layer } from '../engine/topology';
import { NotImplementedError, ValueError } from '../errors';
import { getInitializer, serializeInitializer } from '../initializers';
import { getRegularizer, serializeRegularizer } from '../regularizers';
import * as generic_utils from '../utils/generic_utils';
import * as math_utils from '../utils/math_utils';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';
/**
 * Applies batch normalization on x given mean, var, beta and gamma.
 *
 * I.e. returns:
 *   `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`
 *
 * @param x Input tensor.
 * @param mean Mean of batch.
 * @param variance Variance of batch.
 * @param beta Tensor with which to center the input.
 * @param gamma Tensor by which to scale the input.
 * @param epsilon Fuzz factor.
 * @returns The result of the batch normalization.
 */
export function batchNormalization(x, mean, variance, beta, gamma, epsilon = 1e-3) {
    let out;
    if (x.rank === 2) {
        out = tfc.batchNorm2d(x, mean, variance, beta, gamma, epsilon);
    }
    else if (x.rank === 3) {
        // TODO(cais): Check rank; give proper error message.
        out = tfc.batchNorm3d(x, mean, variance, beta, gamma, epsilon);
    }
    else if (x.rank === 4) {
        out = tfc.batchNorm4d(x, mean, variance, beta, gamma, epsilon);
    }
    else {
        throw new NotImplementedError(`batchNormalization is not implemented for array of rank ${x.rank} ` +
            `yet`);
    }
    return out;
}
/**
 * Non-broadcasting batch normalization for use in training (not inference).
 *
 * The input is normalized to zero mean and unit variance along the
 * `reductionAxes`, followed by scaling with `gamma` and shifted by `beta`.
 * The result of that is returned as the first element
 * of the returned `Array`. The other two elements are the mean and variance,
 * respectively.
 *
 * @param x Input tensor to be normalized.
 * @param gamma Tensor by which to scale the input.
 * @param beta Tensor by which to center the input.
 * @param reductionAxes Axes over which to normalize.
 * @param epsilon Fuzz factor.
 * @returns An `Array` of three `Tensors`:
 *   [normalized tensor, mean of input, variance of input].
 */
function regularNormalizeBatchInTraining(x, gamma, beta, reductionAxes, epsilon = 1e-3) {
    return tidy(() => {
        const meanAndVariance = tfc.moments(x, reductionAxes);
        const mean = meanAndVariance.mean;
        const variance = meanAndVariance.variance;
        const normed = batchNormalization(x, mean, variance, beta, gamma, epsilon);
        return [normed, mean, variance];
    });
}
/**
 * Broadcasting batch normalization for use in training (not inference).
 *
 * The input is normalized to zero mean and unit variance along the
 * `reductionAxes`, followed by scaling with `gamma` and shifted by `beta`.
 * The result of that is returned as the first element
 * of the returned `Array`. The other two elements are the mean and variance,
 * respectively.
 *
 * @param x Input tensor to be normalized.
 * @param gamma Tensor by which to scale the input.
 * @param beta Tensor by which to center the input.
 * @param reductionAxes Axes over which to normalize.
 * @param epsilon Fuzz factor.
 * @returns An `Array` of three `Tensors`:
 *   [normalized tensor, mean of input, variance of input].
 */
function broadcastNormalizeBatchInTraining(x, gamma, beta, reductionAxes, epsilon = 1e-3) {
    return tidy(() => {
        const meanAndVariance = tfc.moments(x, reductionAxes);
        const mean = meanAndVariance.mean;
        const variance = meanAndVariance.variance;
        const targetShape = [];
        for (const axis of math_utils.range(0, x.rank)) {
            if (reductionAxes.indexOf(axis) !== -1) {
                targetShape.push(1);
            }
            else {
                targetShape.push(x.shape[axis]);
            }
        }
        const broadcastMean = reshape(mean, targetShape);
        const broadcastVariance = reshape(variance, targetShape);
        const broadcastGamma = gamma == null ? null : reshape(gamma, targetShape);
        const broadcastBeta = beta == null ? null : reshape(beta, targetShape);
        const normed = batchNormalization(x, broadcastMean, broadcastVariance, broadcastBeta, broadcastGamma, epsilon);
        return [normed, mean, variance];
    });
}
/**
 * Batch normalization for use in training (not inference).
 *
 * @param x Input tensor to be normalized.
 * @param gamma Tensor by which to scale the input.
 * @param beta Tensor by which to center the input.
 * @param reductionAxes Axes over which to normalize.
 * @param epsilon Fuzz factor.
 * @returns An `Array` of three `Tensors`:
 *   [normalized tensor, mean of input, variance of input].
 */
export function normalizeBatchInTraining(x, gamma, beta, reductionAxes, epsilon = 1e-3) {
    if (util.arraysEqual(reductionAxes.slice().sort(), math_utils.range(0, x.rank - 1))) {
        return regularNormalizeBatchInTraining(x, gamma, beta, reductionAxes, epsilon);
    }
    else {
        return broadcastNormalizeBatchInTraining(x, gamma, beta, reductionAxes, epsilon);
    }
}
export class BatchNormalization extends Layer {
    constructor(args) {
        if (args == null) {
            args = {};
        }
        super(args);
        this.supportsMasking = true;
        this.axis = args.axis == null ? -1 : args.axis;
        this.momentum = args.momentum == null ? 0.99 : args.momentum;
        this.epsilon = args.epsilon == null ? 1e-3 : args.epsilon;
        this.center = args.center == null ? true : args.center;
        this.scale = args.scale == null ? true : args.scale;
        this.betaInitializer = getInitializer(args.betaInitializer || 'zeros');
        this.gammaInitializer = getInitializer(args.gammaInitializer || 'ones');
        this.movingMeanInitializer =
            getInitializer(args.movingMeanInitializer || 'zeros');
        this.movingVarianceInitializer =
            getInitializer(args.movingVarianceInitializer || 'ones');
        this.betaConstraint = getConstraint(args.betaConstraint);
        this.gammaConstraint = getConstraint(args.gammaConstraint);
        this.betaRegularizer = getRegularizer(args.betaRegularizer);
        this.gammaRegularizer = getRegularizer(args.gammaRegularizer);
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const axis = this.axis >= 0 ? this.axis : (this.axis + inputShape.length);
        const dim = inputShape[axis];
        if (dim == null) {
            throw new ValueError(`Axis ${axis} of input tensor should have a defined dimension but ` +
                `the layer received an input with shape ` +
                `${JSON.stringify(inputShape)}.`);
        }
        this.inputSpec =
            [new InputSpec({ ndim: inputShape.length, axes: { [axis]: dim } })];
        const shape = [dim];
        if (this.scale) {
            this.gamma = this.addWeight('gamma', shape, null, this.gammaInitializer, this.gammaRegularizer, true, this.gammaConstraint);
        }
        if (this.center) {
            this.beta = this.addWeight('beta', shape, null, this.betaInitializer, this.betaRegularizer, true, this.betaConstraint);
        }
        this.movingMean = this.addWeight('moving_mean', shape, null, this.movingMeanInitializer, null, false);
        this.movingVariance = this.addWeight('moving_variance', shape, null, this.movingVarianceInitializer, null, false);
        this.built = true;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            const training = kwargs['training'] == null ? false : kwargs['training'];
            const input = getExactlyOneTensor(inputs);
            const inputShape = input.shape;
            const ndim = inputShape.length;
            const reductionAxes = math_utils.range(0, ndim);
            const axis = this.axis >= 0 ? this.axis : (this.axis + ndim);
            reductionAxes.splice(axis, 1);
            const broadcastShape = generic_utils.pyListRepeat(1, ndim);
            broadcastShape[axis] = inputShape[axis];
            const sortedReductionAxes = reductionAxes.slice();
            sortedReductionAxes.sort();
            const needsBroadcasting = !util.arraysEqual(sortedReductionAxes, math_utils.range(0, ndim).slice(0, ndim - 1));
            const normalizeInference = () => {
                if (needsBroadcasting) {
                    const broadcastMovingMean = reshape(this.movingMean.read(), broadcastShape);
                    const broadcastMovingVariance = reshape(this.movingVariance.read(), broadcastShape);
                    const broadcastBeta = this.center ? reshape(this.beta.read(), broadcastShape) : null;
                    const broadcastGamma = this.scale ? reshape(this.gamma.read(), broadcastShape) : null;
                    return batchNormalization(input, broadcastMovingMean, broadcastMovingVariance, broadcastBeta, broadcastGamma, this.epsilon);
                }
                else {
                    return batchNormalization(input, this.movingMean.read(), this.movingVariance.read(), this.beta == null ? null : this.beta.read(), this.gamma == null ? null : this.gamma.read(), this.epsilon);
                }
            };
            if (!training) {
                return normalizeInference();
            }
            const [normedTraining, mean, variance] = normalizeBatchInTraining(input, this.gamma.read(), this.beta.read(), reductionAxes, this.epsilon);
            const doMovingAverage = (variable, value, momentum) => {
                tfc.tidy(() => {
                    const decay = 1 - momentum;
                    const origValue = variable.read();
                    const updateDelta = tfc.mul(tfc.sub(origValue, value), decay);
                    variable.write(tfc.sub(origValue, updateDelta));
                });
            };
            // Perform updates to moving mean and moving variance for training.
            // Porting Note: In PyKeras, these updates to `movingMean` and
            //   `movingAverage` are done as a deferred Graph, added to the `Layer`'s
            //   `update`s using the `add_update()` method. Here we do it imperatively
            //   and encapsulate the updates in a function that is invoked
            //   immediately.
            const updateMovingMeanAndVariance = () => {
                doMovingAverage(this.movingMean, mean, this.momentum);
                doMovingAverage(this.movingVariance, variance, this.momentum);
            };
            updateMovingMeanAndVariance();
            return normedTraining;
        });
    }
    getConfig() {
        const config = {
            axis: this.axis,
            momentum: this.momentum,
            epsilon: this.epsilon,
            center: this.center,
            scale: this.scale,
            betaInitializer: serializeInitializer(this.betaInitializer),
            gammaInitializer: serializeInitializer(this.gammaInitializer),
            movingMeanInitializer: serializeInitializer(this.movingMeanInitializer),
            movingVarianceInitializer: serializeInitializer(this.movingVarianceInitializer),
            betaRegularizer: serializeRegularizer(this.betaRegularizer),
            gammaRegularizer: serializeRegularizer(this.gammaRegularizer),
            betaConstraint: serializeConstraint(this.betaConstraint),
            gammaConstraint: serializeConstraint(this.gammaConstraint)
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
BatchNormalization.className = 'BatchNormalization';
serialization.registerClass(BatchNormalization);
export class LayerNormalization extends Layer {
    constructor(args) {
        if (args == null) {
            args = {};
        }
        super(args);
        this.axis = args.axis == null ? -1 : args.axis;
        if (typeof this.axis === 'number') {
            if (!Number.isInteger(this.axis)) {
                throw new Error(`Expected axis to be an integer, but received ${this.axis}`);
            }
        }
        else if (Array.isArray(this.axis)) {
            for (const axis of this.axis) {
                if (!Number.isInteger(axis)) {
                    throw new Error(`Expected axis to be an array of integers, ` +
                        `but received ${JSON.stringify(this.axis)}`);
                }
            }
        }
        else {
            throw new Error(`Expected axis to be an integer or an array of integers, ` +
                `but received ${JSON.stringify(this.axis)}`);
        }
        this.epsilon = args.epsilon == null ? 1e-3 : args.epsilon;
        this.center = args.center == null ? true : args.center;
        this.scale = args.scale == null ? true : args.scale;
        this.betaInitializer = getInitializer(args.betaInitializer || 'zeros');
        this.gammaInitializer = getInitializer(args.gammaInitializer || 'ones');
        this.betaRegularizer = getRegularizer(args.betaRegularizer);
        this.gammaRegularizer = getRegularizer(args.gammaRegularizer);
        this.supportsMasking = true;
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const nDims = inputShape.length;
        // Convert axis to array and resolve negatives.
        if (typeof this.axis === 'number') {
            this.axis = [this.axis];
        }
        for (let i = 0; i < this.axis.length; ++i) {
            if (this.axis[i] < 0) {
                this.axis[i] += nDims;
            }
        }
        // Further validate axes.
        for (const axis of this.axis) {
            if (axis < 0 || axis >= nDims) {
                throw new Error(`Invalid axis: ${axis}`);
            }
        }
        if (this.axis.length !== generic_utils.unique(this.axis).length) {
            throw new Error(`Found duplicate axes in: ${this.axis}`);
        }
        const paramShape = this.axis.map(axis => inputShape[axis]);
        const trainable = true;
        if (this.scale) {
            this.gamma = this.addWeight('gamma', paramShape, 'float32', this.gammaInitializer, this.gammaRegularizer, trainable);
        }
        else {
            this.gamma = null;
        }
        if (this.center) {
            this.beta = this.addWeight('beta', paramShape, 'float32', this.betaInitializer, this.betaRegularizer, trainable);
        }
        else {
            this.beta = null;
        }
        this.built = true;
    }
    call(inputs, kwargs) {
        const input = getExactlyOneTensor(inputs);
        const inputShape = input.shape;
        const nDims = inputShape.length;
        return tidy(() => {
            const keepDims = true;
            let { mean, variance } = moments(input, this.axis, keepDims);
            const broadcastShape = generic_utils.pyListRepeat(1, nDims);
            for (const dim of this.axis) {
                broadcastShape[dim] = inputShape[dim];
            }
            const broadcast = (v) => {
                if (v != null && v.shape.length !== nDims) {
                    return tfc.reshape(v, broadcastShape);
                }
                else {
                    return v;
                }
            };
            let scale = this.scale ? broadcast(this.gamma.read()) : null;
            let offset = this.center ? broadcast(this.beta.read()) : null;
            // TODO(https://github.com/tensorflow/tfjs/issues/2120): The tiling below
            // is a workaround for the limitation of core's batchNormalization?d don't
            // support broadcasting in their gradients. In addition, the tiling is
            // necessary to ensure correctness on the browser CPU backend regardless
            // of forward or backward computation. Remove this workaround once the
            // limitation is addressed. See .
            const momentsTiling = [];
            const scaleOffsetTiling = [];
            for (let i = 0; i < nDims; ++i) {
                if (this.axis.indexOf(i) !== -1) {
                    momentsTiling.push(inputShape[i]);
                    scaleOffsetTiling.push(1);
                }
                else {
                    momentsTiling.push(1);
                    scaleOffsetTiling.push(inputShape[i]);
                }
            }
            mean = tfc.tile(mean, momentsTiling);
            variance = tfc.tile(variance, momentsTiling);
            if (scale != null) {
                scale = tfc.tile(scale, scaleOffsetTiling);
            }
            if (offset != null) {
                offset = tfc.tile(offset, scaleOffsetTiling);
            }
            return batchNormalization(input, mean, variance, offset, scale, this.epsilon);
        });
    }
    getConfig() {
        const config = {
            axis: this.axis,
            epsilon: this.epsilon,
            center: this.center,
            scale: this.scale,
            betaInitializer: serializeInitializer(this.betaInitializer),
            gammaInitializer: serializeInitializer(this.gammaInitializer),
            betaRegularizer: serializeRegularizer(this.betaRegularizer),
            gammaRegularizer: serializeRegularizer(this.gammaRegularizer)
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
LayerNormalization.className = 'LayerNormalization';
serialization.registerClass(LayerNormalization);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibm9ybWFsaXphdGlvbi5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9sYXllcnMvbm9ybWFsaXphdGlvbi50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVIOztHQUVHO0FBRUgsT0FBTyxLQUFLLEdBQUcsTUFBTSx1QkFBdUIsQ0FBQztBQUM3QyxPQUFPLEVBQUMsT0FBTyxFQUFFLE9BQU8sRUFBRSxhQUFhLEVBQWtELElBQUksRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUVsSSxPQUFPLEVBQW1DLGFBQWEsRUFBRSxtQkFBbUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQ3BHLE9BQU8sRUFBQyxTQUFTLEVBQUUsS0FBSyxFQUFZLE1BQU0sb0JBQW9CLENBQUM7QUFDL0QsT0FBTyxFQUFDLG1CQUFtQixFQUFFLFVBQVUsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUMxRCxPQUFPLEVBQUMsY0FBYyxFQUFzQyxvQkFBb0IsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBRXpHLE9BQU8sRUFBQyxjQUFjLEVBQXNDLG9CQUFvQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFekcsT0FBTyxLQUFLLGFBQWEsTUFBTSx3QkFBd0IsQ0FBQztBQUN4RCxPQUFPLEtBQUssVUFBVSxNQUFNLHFCQUFxQixDQUFDO0FBQ2xELE9BQU8sRUFBQyxrQkFBa0IsRUFBRSxtQkFBbUIsRUFBQyxNQUFNLHNCQUFzQixDQUFDO0FBRzdFOzs7Ozs7Ozs7Ozs7O0dBYUc7QUFDSCxNQUFNLFVBQVUsa0JBQWtCLENBQzlCLENBQVMsRUFBRSxJQUFZLEVBQUUsUUFBZ0IsRUFBRSxJQUFhLEVBQUUsS0FBYyxFQUN4RSxPQUFPLEdBQUcsSUFBSTtJQUNoQixJQUFJLEdBQVcsQ0FBQztJQUNoQixJQUFJLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ2hCLEdBQUcsR0FB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O1FBQ2hCLE1BQU0sTUFBTSxHQUE2QjtZQUN2QyxJQUFJLEVBQUUsSUFBSSxDQUFDLElBQUk7WUFDZixPQUFPLEVBQUUsSUFBSSxDQUFDLE9BQU87WUFDckIsTUFBTSxFQUFFLElBQUksQ0FBQyxNQUFNO1lBQ25CLEtBQUssRUFBRSxJQUFJLENBQUMsS0FBSztZQUNqQixlQUFlLEVBQUUsb0JBQW9CLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQztZQUMzRCxnQkFBZ0IsRUFBRSxvQkFBb0IsQ0FBQyxJQUFJLENBQUMsZ0JBQWdCLENBQUM7WUFDN0QsZUFBZSxFQUFFLG9CQUFvQixDQUFDLElBQUksQ0FBQyxlQUFlLENBQUM7WUFDM0QsZ0JBQWdCLEVBQUUsb0JBQW9CLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDO1NBQzlELENBQUM7UUFDRixNQUFNLFVBQVUsR0FBRyxLQUFLLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDckMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxNQUFNLEVBQUUsVUFBVSxDQUFDLENBQUM7UUFDbEMsT0FBTyxNQUFNLENBQUM7SUFDaEIsQ0FBQzs7QUF0S0Qsa0JBQWtCO0FBQ1gsNEJBQVMsR0FBRyxvQkFBb0IsQ0FBQztBQXVLMUMsYUFBYSxDQUFDLGFBQWEsQ0FBQyxrQkFBa0IsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQ1xuICpcbiAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZVxuICogbGljZW5zZSB0aGF0IGNhbiBiZSBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIG9yIGF0XG4gKiBodHRwczovL29wZW5zb3VyY2Uub3JnL2xpY2Vuc2VzL01JVC5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyoqXG4gKiBOb3JtYWxpemF0aW9uIGxheWVycy5cbiAqL1xuXG5pbXBvcnQgKiBhcyB0ZmMgZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7bW9tZW50cywgcmVzaGFwZSwgc2VyaWFsaXphdGlvbiwgVGVuc29yLCBUZW5zb3IxRCwgVGVuc29yMkQsIFRlbnNvcjNELCBUZW5zb3I0RCwgdGlkeSwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtDb25zdHJhaW50LCBDb25zdHJhaW50SWRlbnRpZmllciwgZ2V0Q29uc3RyYWludCwgc2VyaWFsaXplQ29uc3RyYWludH0gZnJvbSAnLi4vY29uc3RyYWludHMnO1xuaW1wb3J0IHtJbnB1dFNwZWMsIExheWVyLCBMYXllckFyZ3N9IGZyb20gJy4uL2VuZ2luZS90b3BvbG9neSc7XG5pbXBvcnQge05vdEltcGxlbWVudGVkRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4uL2Vycm9ycyc7XG5pbXBvcnQge2dldEluaXRpYWxpemVyLCBJbml0aWFsaXplciwgSW5pdGlhbGl6ZXJJZGVudGlmaWVyLCBzZXJpYWxpemVJbml0aWFsaXplcn0gZnJvbSAnLi4vaW5pdGlhbGl6ZXJzJztcbmltcG9ydCB7U2hhcGV9IGZyb20gJy4uL2tlcmFzX2Zvcm1hdC9jb21tb24nO1xuaW1wb3J0IHtnZXRSZWd1bGFyaXplciwgUmVndWxhcml6ZXIsIFJlZ3VsYXJpemVySWRlbnRpZmllciwgc2VyaWFsaXplUmVndWxhcml6ZXJ9IGZyb20gJy4uL3JlZ3VsYXJpemVycyc7XG5pbXBvcnQge0t3YXJnc30gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgZ2VuZXJpY191dGlscyBmcm9tICcuLi91dGlscy9nZW5lcmljX3V0aWxzJztcbmltcG9ydCAqIGFzIG1hdGhfdXRpbHMgZnJvbSAnLi4vdXRpbHMvbWF0aF91dGlscyc7XG5pbXBvcnQge2dldEV4YWN0bHlPbmVTaGFwZSwgZ2V0RXhhY3RseU9uZVRlbnNvcn0gZnJvbSAnLi4vdXRpbHMvdHlwZXNfdXRpbHMnO1xuaW1wb3J0IHtMYXllclZhcmlhYmxlfSBmcm9tICcuLi92YXJpYWJsZXMnO1xuXG4vKipcbiAqIEFwcGxpZXMgYmF0Y2ggbm9ybWFsaXphdGlvbiBvbiB4IGdpdmVuIG1lYW4sIHZhciwgYmV0YSBhbmQgZ2FtbWEuXG4gKlxuICogSS5lLiByZXR1cm5zOlxuICogICBgb3V0cHV0ID0gKHggLSBtZWFuKSAvIChzcXJ0KHZhcikgKyBlcHNpbG9uKSAqIGdhbW1hICsgYmV0YWBcbiAqXG4gKiBAcGFyYW0geCBJbnB1dCB0ZW5zb3IuXG4gKiBAcGFyYW0gbWVhbiBNZWFuIG9mIGJhdGNoLlxuICogQHBhcmFtIHZhcmlhbmNlIFZhcmlhbmNlIG9mIGJhdGNoLlxuICogQHBhcmFtIGJldGEgVGVuc29yIHdpdGggd2hpY2ggdG8gY2VudGVyIHRoZSBpbnB1dC5cbiAqIEBwYXJhbSBnYW1tYSBUZW5zb3IgYnkgd2hpY2ggdG8gc2NhbGUgdGhlIGlucHV0LlxuICogQHBhcmFtIGVwc2lsb24gRnV6eiBmYWN0b3IuXG4gKiBAcmV0dXJucyBUaGUgcmVzdWx0IG9mIHRoZSBiYXRjaCBub3JtYWxpemF0aW9uLlxuICovXG5leHBvcnQgZnVuY3Rpb24gYmF0Y2hOb3JtYWxpemF0aW9uKFxuICAgIHg6IFRlbnNvciwgbWVhbjogVGVuc29yLCB2YXJpYW5jZTogVGVuc29yLCBiZXRhPzogVGVuc29yLCBnYW1tYT86IFRlbnNvcixcbiAgICBlcHNpbG9uID0gMWUtMyk6IFRlbnNvciB7XG4gIGxldCBvdXQ6IFRlbnNvcjtcbiAgaWYgKHgucmFuayA9PT0gMikge1xuICAgIG91dCA9IHRmYy5iYXRjaE5vcm0yZChcbiAgICAgICAgeCBhcyBUZW5zb3IyRCwgbWVhbiBhcyBUZW5zb3IyRCB8IFRlbnNvcjFELFxuICAgICAgICB2YXJpYW5jZSBhcyBUZW5zb3IyRCB8IFRlbnNvcjFELCBiZXRhIGFzIFRlbnNvcjJEIHwgVGVuc29yMUQsXG4gICAgICAgIGdhbW1hIGFzIFRlbnNvcjJEIHwgVGVuc29yMUQsIGVwc2lsb24pO1xuICB9IGVsc2UgaWYgKHgucmFuayA9PT0gMykge1xuICAgIC8vIFRPRE8oY2Fpcyk6IENoZWNrIHJhbms7IGdpdmUgcHJvcGVyIGVycm9yIG1lc3NhZ2UuXG4gICAgb3V0ID0gdGZjLmJhdGNoTm9ybTNkKFxuICAgICAgICB4IGFzIFRlbnNvcjNELCBtZWFuIGFzIFRlbnNvcjNEIHwgVGVuc29yMUQsXG4gICAgICAgIHZhcmlhbmNlIGFzIFRlbnNvcjNEIHwgVGVuc29yMUQsIGJldGEgYXMgVGVuc29yM0QgfCBUZW5zb3IxRCxcbiAgICAgICAgZ2FtbWEgYXMgVGVuc29yM0QgfCBUZW5zb3IxRCwgZXBzaWxvbik7XG4gIH0gZWxzZSBpZiAoeC5yYW5rID09PSA0KSB7XG4gICAgb3V0ID0gdGZjLmJhdGNoTm9ybTRkKFxuICAgICAgICB4IGFzIFRlbnNvcjRELCBtZWFuIGFzIFRlbnNvcjREIHwgVGVuc29yMUQsXG4gICAgICAgIHZhcmlhbmNlIGFzIFRlbnNvcjREIHwgVGVuc29yMUQsIGJldGEgYXMgVGVuc29yNEQgfCBUZW5zb3IxRCxcbiAgICAgICAgZ2FtbWEgYXMgVGVuc29yNEQgfCBUZW5zb3IxRCwgZXBzaWxvbik7XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgIGBiYXRjaE5vcm1hbGl6YXRpb24gaXMgbm90IGltcGxlbWVudGVkIGZvciBhcnJheSBvZiByYW5rICR7eC5yYW5rfSBgICtcbiAgICAgICAgYHlldGApO1xuICB9XG4gIHJldHVybiBvdXQ7XG59XG5cbi8qKlxuICogTm9uLWJyb2FkY2FzdGluZyBiYXRjaCBub3JtYWxpemF0aW9uIGZvciB1c2UgaW4gdHJhaW5pbmcgKG5vdCBpbmZlcmVuY2UpLlxuICpcbiAqIFRoZSBpbnB1dCBpcyBub3JtYWxpemVkIHRvIHplcm8gbWVhbiBhbmQgdW5pdCB2YXJpYW5jZSBhbG9uZyB0aGVcbiAqIGByZWR1Y3Rpb25BeGVzYCwgZm9sbG93ZWQgYnkgc2NhbGluZyB3aXRoIGBnYW1tYWAgYW5kIHNoaWZ0ZWQgYnkgYGJldGFgLlxuICogVGhlIHJlc3VsdCBvZiB0aGF0IGlzIHJldHVybmVkIGFzIHRoZSBmaXJzdCBlbGVtZW50XG4gKiBvZiB0aGUgcmV0dXJuZWQgYEFycmF5YC4gVGhlIG90aGVyIHR3byBlbGVtZW50cyBhcmUgdGhlIG1lYW4gYW5kIHZhcmlhbmNlLFxuICogcmVzcGVjdGl2ZWx5LlxuICpcbiAqIEBwYXJhbSB4IElucHV0IHRlbnNvciB0byBiZSBub3JtYWxpemVkLlxuICogQHBhcmFtIGdhbW1hIFRlbnNvciBieSB3aGljaCB0byBzY2FsZSB0aGUgaW5wdXQuXG4gKiBAcGFyYW0gYmV0YSBUZW5zb3IgYnkgd2hpY2ggdG8gY2VudGVyIHRoZSBpbnB1dC5cbiAqIEBwYXJhbSByZWR1Y3Rpb25BeGVzIEF4ZXMgb3ZlciB3aGljaCB0byBub3JtYWxpemUuXG4gKiBAcGFyYW0gZXBzaWxvbiBGdXp6IGZhY3Rvci5cbiAqIEByZXR1cm5zIEFuIGBBcnJheWAgb2YgdGhyZWUgYFRlbnNvcnNgOlxuICogICBbbm9ybWFsaXplZCB0ZW5zb3IsIG1lYW4gb2YgaW5wdXQsIHZhcmlhbmNlIG9mIGlucHV0XS5cbiAqL1xuZnVuY3Rpb24gcmVndWxhck5vcm1hbGl6ZUJhdGNoSW5UcmFpbmluZyhcbiAgICB4OiBUZW5zb3IsIGdhbW1hOiBUZW5zb3IsIGJldGE6IFRlbnNvciwgcmVkdWN0aW9uQXhlczogbnVtYmVyW10sXG4gICAgZXBzaWxvbiA9IDFlLTMpOiBbVGVuc29yLCBUZW5zb3IsIFRlbnNvcl0ge1xuICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICAgICAgIGNvbnN0IG1lYW5BbmRWYXJpYW5jZSA9IHRmYy5tb21lbnRzKHgsIHJlZHVjdGlvbkF4ZXMpO1xuICAgICAgICAgICBjb25zdCBtZWFuID0gbWVhbkFuZFZhcmlhbmNlLm1lYW47XG4gICAgICAgICAgIGNvbnN0IHZhcmlhbmNlID0gbWVhbkFuZFZhcmlhbmNlLnZhcmlhbmNlO1xuICAgICAgICAgICBjb25zdCBub3JtZWQgPVxuICAgICAgICAgICAgICAgYmF0Y2hOb3JtYWxpemF0aW9uKHgsIG1lYW4sIHZhcmlhbmNlLCBiZXRhLCBnYW1tYSwgZXBzaWxvbik7XG4gICAgICAgICAgIHJldHVybiBbbm9ybWVkLCBtZWFuLCB2YXJpYW5jZV07XG4gICAgICAgICB9KSBhcyBbVGVuc29yLCBUZW5zb3IsIFRlbnNvcl07XG59XG5cbi8qKlxuICogQnJvYWRjYXN0aW5nIGJhdGNoIG5vcm1hbGl6YXRpb24gZm9yIHVzZSBpbiB0cmFpbmluZyAobm90IGluZmVyZW5jZSkuXG4gKlxuICogVGhlIGlucHV0IGlzIG5vcm1hbGl6ZWQgdG8gemVybyBtZWFuIGFuZCB1bml0IHZhcmlhbmNlIGFsb25nIHRoZVxuICogYHJlZHVjdGlvbkF4ZXNgLCBmb2xsb3dlZCBieSBzY2FsaW5nIHdpdGggYGdhbW1hYCBhbmQgc2hpZnRlZCBieSBgYmV0YWAuXG4gKiBUaGUgcmVzdWx0IG9mIHRoYXQgaXMgcmV0dXJuZWQgYXMgdGhlIGZpcnN0IGVsZW1lbnRcbiAqIG9mIHRoZSByZXR1cm5lZCBgQXJyYXlgLiBUaGUgb3RoZXIgdHdvIGVsZW1lbnRzIGFyZSB0aGUgbWVhbiBhbmQgdmFyaWFuY2UsXG4gKiByZXNwZWN0aXZlbHkuXG4gKlxuICogQHBhcmFtIHggSW5wdXQgdGVuc29yIHRvIGJlIG5vcm1hbGl6ZWQuXG4gKiBAcGFyYW0gZ2FtbWEgVGVuc29yIGJ5IHdoaWNoIHRvIHNjYWxlIHRoZSBpbnB1dC5cbiAqIEBwYXJhbSBiZXRhIFRlbnNvciBieSB3aGljaCB0byBjZW50ZXIgdGhlIGlucHV0LlxuICogQHBhcmFtIHJlZHVjdGlvbkF4ZXMgQXhlcyBvdmVyIHdoaWNoIHRvIG5vcm1hbGl6ZS5cbiAqIEBwYXJhbSBlcHNpbG9uIEZ1enogZmFjdG9yLlxuICogQHJldHVybnMgQW4gYEFycmF5YCBvZiB0aHJlZSBgVGVuc29yc2A6XG4gKiAgIFtub3JtYWxpemVkIHRlbnNvciwgbWVhbiBvZiBpbnB1dCwgdmFyaWFuY2Ugb2YgaW5wdXRdLlxuICovXG5mdW5jdGlvbiBicm9hZGNhc3ROb3JtYWxpemVCYXRjaEluVHJhaW5pbmcoXG4gICAgeDogVGVuc29yLCBnYW1tYTogVGVuc29yLCBiZXRhOiBUZW5zb3IsIHJlZHVjdGlvbkF4ZXM6IG51bWJlcltdLFxuICAgIGVwc2lsb24gPSAxZS0zKTogW1RlbnNvciwgVGVuc29yLCBUZW5zb3JdIHtcbiAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgICAgICBjb25zdCBtZWFuQW5kVmFyaWFuY2UgPSB0ZmMubW9tZW50cyh4LCByZWR1Y3Rpb25BeGVzKTtcbiAgICAgICAgICAgY29uc3QgbWVhbiA9IG1lYW5BbmRWYXJpYW5jZS5tZWFuO1xuICAgICAgICAgICBjb25zdCB2YXJpYW5jZSA9IG1lYW5BbmRWYXJpYW5jZS52YXJpYW5jZTtcbiAgICAgICAgICAgY29uc3QgdGFyZ2V0U2hhcGU6IG51bWJlcltdID0gW107XG4gICAgICAgICAgIGZvciAoY29uc3QgYXhpcyBvZiBtYXRoX3V0aWxzLnJhbmdlKDAsIHgucmFuaykpIHtcbiAgICAgICAgICAgICBpZiAocmVkdWN0aW9uQXhlcy5pbmRleE9mKGF4aXMpICE9PSAtMSkge1xuICAgICAgICAgICAgICAgdGFyZ2V0U2hhcGUucHVzaCgxKTtcbiAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgdGFyZ2V0U2hhcGUucHVzaCh4LnNoYXBlW2F4aXNdKTtcbiAgICAgICAgICAgICB9XG4gICAgICAgICAgIH1cbiAgICAgICAgICAgY29uc3QgYnJvYWRjYXN0TWVhbiA9IHJlc2hhcGUobWVhbiwgdGFyZ2V0U2hhcGUpO1xuICAgICAgICAgICBjb25zdCBicm9hZGNhc3RWYXJpYW5jZSA9IHJlc2hhcGUodmFyaWFuY2UsIHRhcmdldFNoYXBlKTtcbiAgICAgICAgICAgY29uc3QgYnJvYWRjYXN0R2FtbWEgPVxuICAgICAgICAgICAgICAgZ2FtbWEgPT0gbnVsbCA/IG51bGwgOiByZXNoYXBlKGdhbW1hLCB0YXJnZXRTaGFwZSk7XG4gICAgICAgICAgIGNvbnN0IGJyb2FkY2FzdEJldGEgPVxuICAgICAgICAgICAgICAgYmV0YSA9PSBudWxsID8gbnVsbCA6IHJlc2hhcGUoYmV0YSwgdGFyZ2V0U2hhcGUpO1xuICAgICAgICAgICBjb25zdCBub3JtZWQgPSBiYXRjaE5vcm1hbGl6YXRpb24oXG4gICAgICAgICAgICAgICB4LCBicm9hZGNhc3RNZWFuLCBicm9hZGNhc3RWYXJpYW5jZSwgYnJvYWRjYXN0QmV0YSxcbiAgICAgICAgICAgICAgIGJyb2FkY2FzdEdhbW1hLCBlcHNpbG9uKTtcbiAgICAgICAgICAgcmV0dXJuIFtub3JtZWQsIG1lYW4sIHZhcmlhbmNlXTtcbiAgICAgICAgIH0pIGFzIFtUZW5zb3IsIFRlbnNvciwgVGVuc29yXTtcbn1cblxuLyoqXG4gKiBCYXRjaCBub3JtYWxpemF0aW9uIGZvciB1c2UgaW4gdHJhaW5pbmcgKG5vdCBpbmZlcmVuY2UpLlxuICpcbiAqIEBwYXJhbSB4IElucHV0IHRlbnNvciB0byBiZSBub3JtYWxpemVkLlxuICogQHBhcmFtIGdhbW1hIFRlbnNvciBieSB3aGljaCB0byBzY2FsZSB0aGUgaW5wdXQuXG4gKiBAcGFyYW0gYmV0YSBUZW5zb3IgYnkgd2hpY2ggdG8gY2VudGVyIHRoZSBpbnB1dC5cbiAqIEBwYXJhbSByZWR1Y3Rpb25BeGVzIEF4ZXMgb3ZlciB3aGljaCB0byBub3JtYWxpemUuXG4gKiBAcGFyYW0gZXBzaWxvbiBGdXp6IGZhY3Rvci5cbiAqIEByZXR1cm5zIEFuIGBBcnJheWAgb2YgdGhyZWUgYFRlbnNvcnNgOlxuICogICBbbm9ybWFsaXplZCB0ZW5zb3IsIG1lYW4gb2YgaW5wdXQsIHZhcmlhbmNlIG9mIGlucHV0XS5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG5vcm1hbGl6ZUJhdGNoSW5UcmFpbmluZyhcbiAgICB4OiBUZW5zb3IsIGdhbW1hOiBUZW5zb3IsIGJldGE6IFRlbnNvciwgcmVkdWN0aW9uQXhlczogbnVtYmVyW10sXG4gICAgZXBzaWxvbiA9IDFlLTMpOiBbVGVuc29yLCBUZW5zb3IsIFRlbnNvcl0ge1xuICBpZiAodXRpbC5hcnJheXNFcXVhbChcbiAgICAgICAgICByZWR1Y3Rpb25BeGVzLnNsaWNlKCkuc29ydCgpLCBtYXRoX3V0aWxzLnJhbmdlKDAsIHgucmFuayAtIDEpKSkge1xuICAgIHJldHVybiByZWd1bGFyTm9ybWFsaXplQmF0Y2hJblRyYWluaW5nKFxuICAgICAgICB4LCBnYW1tYSwgYmV0YSwgcmVkdWN0aW9uQXhlcywgZXBzaWxvbik7XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIGJyb2FkY2FzdE5vcm1hbGl6ZUJhdGNoSW5UcmFpbmluZyhcbiAgICAgICAgeCwgZ2FtbWEsIGJldGEsIHJlZHVjdGlvbkF4ZXMsIGVwc2lsb24pO1xuICB9XG59XG5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBCYXRjaE5vcm1hbGl6YXRpb25MYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogVGhlIGludGVnZXIgYXhpcyB0aGF0IHNob3VsZCBiZSBub3JtYWxpemVkICh0eXBpY2FsbHkgdGhlIGZlYXR1cmVzIGF4aXMpLlxuICAgKiBEZWZhdWx0cyB0byAtMS5cbiAgICpcbiAgICogRm9yIGluc3RhbmNlLCBhZnRlciBhIGBDb252MkRgIGxheWVyIHdpdGggYGRhdGFfZm9ybWF0PVwiY2hhbm5lbHNfZmlyc3RcImAsXG4gICAqIHNldCBgYXhpcz0xYCBpbiBgYmF0Y2hOb3JtYWxpemF0aW9uYC5cbiAgICovXG4gIGF4aXM/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE1vbWVudHVtIG9mIHRoZSBtb3ZpbmcgYXZlcmFnZS4gRGVmYXVsdHMgdG8gMC45OS5cbiAgICovXG4gIG1vbWVudHVtPzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBTbWFsbCBmbG9hdCBhZGRlZCB0byB0aGUgdmFyaWFuY2UgdG8gYXZvaWQgZGl2aWRpbmcgYnkgemVyby4gRGVmYXVsdHMgdG9cbiAgICogMWUtMy5cbiAgICovXG4gIGVwc2lsb24/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIElmIGB0cnVlYCwgYWRkIG9mZnNldCBvZiBgYmV0YWAgdG8gbm9ybWFsaXplZCB0ZW5zb3IuXG4gICAqIElmIGBmYWxzZWAsIGBiZXRhYCBpcyBpZ25vcmVkLlxuICAgKiBEZWZhdWx0cyB0byBgdHJ1ZWAuXG4gICAqL1xuICBjZW50ZXI/OiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBJZiBgdHJ1ZWAsIG11bHRpcGx5IGJ5IGBnYW1tYWAuXG4gICAqIElmIGBmYWxzZWAsIGBnYW1tYWAgaXMgbm90IHVzZWQuXG4gICAqIFdoZW4gdGhlIG5leHQgbGF5ZXIgaXMgbGluZWFyIChhbHNvIGUuZy4gYG5uLnJlbHVgKSxcbiAgICogdGhpcyBjYW4gYmUgZGlzYWJsZWQgc2luY2UgdGhlIHNjYWxpbmcgd2lsbCBiZSBkb25lIGJ5IHRoZSBuZXh0IGxheWVyLlxuICAgKiBEZWZhdWx0cyB0byBgdHJ1ZWAuXG4gICAqL1xuICBzY2FsZT86IGJvb2xlYW47XG5cbiAgLyoqXG4gICAqIEluaXRpYWxpemVyIGZvciB0aGUgYmV0YSB3ZWlnaHQuXG4gICAqICBEZWZhdWx0cyB0byAnemVyb3MnLlxuICAgKi9cbiAgYmV0YUluaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyO1xuXG4gIC8qKlxuICAgKiBJbml0aWFsaXplciBmb3IgdGhlIGdhbW1hIHdlaWdodC5cbiAgICogIERlZmF1bHRzIHRvIGBvbmVzYC5cbiAgICovXG4gIGdhbW1hSW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcklkZW50aWZpZXJ8SW5pdGlhbGl6ZXI7XG5cbiAgLyoqXG4gICAqIEluaXRpYWxpemVyIGZvciB0aGUgbW92aW5nIG1lYW4uXG4gICAqIERlZmF1bHRzIHRvIGB6ZXJvc2BcbiAgICovXG4gIG1vdmluZ01lYW5Jbml0aWFsaXplcj86IEluaXRpYWxpemVySWRlbnRpZmllcnxJbml0aWFsaXplcjtcblxuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBtb3ZpbmcgdmFyaWFuY2UuXG4gICAqICBEZWZhdWx0cyB0byAnT25lcycuXG4gICAqL1xuICBtb3ZpbmdWYXJpYW5jZUluaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyO1xuXG4gIC8qKlxuICAgKiBDb25zdHJhaW50IGZvciB0aGUgYmV0YSB3ZWlnaHQuXG4gICAqL1xuICBiZXRhQ29uc3RyYWludD86IENvbnN0cmFpbnRJZGVudGlmaWVyfENvbnN0cmFpbnQ7XG5cbiAgLyoqXG4gICAqIENvbnN0cmFpbnQgZm9yIGdhbW1hIHdlaWdodC5cbiAgICovXG4gIGdhbW1hQ29uc3RyYWludD86IENvbnN0cmFpbnRJZGVudGlmaWVyfENvbnN0cmFpbnQ7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZvciB0aGUgYmV0YSB3ZWlnaHQuXG4gICAqL1xuICBiZXRhUmVndWxhcml6ZXI/OiBSZWd1bGFyaXplcklkZW50aWZpZXJ8UmVndWxhcml6ZXI7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZvciB0aGUgZ2FtbWEgd2VpZ2h0LlxuICAgKi9cbiAgZ2FtbWFSZWd1bGFyaXplcj86IFJlZ3VsYXJpemVySWRlbnRpZmllcnxSZWd1bGFyaXplcjtcbn1cblxuZXhwb3J0IGNsYXNzIEJhdGNoTm9ybWFsaXphdGlvbiBleHRlbmRzIExheWVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnQmF0Y2hOb3JtYWxpemF0aW9uJztcbiAgcHJpdmF0ZSByZWFkb25seSBheGlzOiBudW1iZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgbW9tZW50dW06IG51bWJlcjtcbiAgcHJpdmF0ZSByZWFkb25seSBlcHNpbG9uOiBudW1iZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgY2VudGVyOiBib29sZWFuO1xuICBwcml2YXRlIHJlYWRvbmx5IHNjYWxlOiBib29sZWFuO1xuICBwcml2YXRlIHJlYWRvbmx5IGJldGFJbml0aWFsaXplcjogSW5pdGlhbGl6ZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgZ2FtbWFJbml0aWFsaXplcjogSW5pdGlhbGl6ZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgbW92aW5nTWVhbkluaXRpYWxpemVyOiBJbml0aWFsaXplcjtcbiAgcHJpdmF0ZSByZWFkb25seSBtb3ZpbmdWYXJpYW5jZUluaXRpYWxpemVyOiBJbml0aWFsaXplcjtcbiAgcHJpdmF0ZSByZWFkb25seSBiZXRhQ29uc3RyYWludDogQ29uc3RyYWludDtcbiAgcHJpdmF0ZSByZWFkb25seSBnYW1tYUNvbnN0cmFpbnQ6IENvbnN0cmFpbnQ7XG4gIHByaXZhdGUgcmVhZG9ubHkgYmV0YVJlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcbiAgcHJpdmF0ZSByZWFkb25seSBnYW1tYVJlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcbiAgcHJpdmF0ZSBnYW1tYTogTGF5ZXJWYXJpYWJsZTtcbiAgcHJpdmF0ZSBiZXRhOiBMYXllclZhcmlhYmxlO1xuICBwcml2YXRlIG1vdmluZ01lYW46IExheWVyVmFyaWFibGU7XG4gIHByaXZhdGUgbW92aW5nVmFyaWFuY2U6IExheWVyVmFyaWFibGU7XG5cbiAgY29uc3RydWN0b3IoYXJncz86IEJhdGNoTm9ybWFsaXphdGlvbkxheWVyQXJncykge1xuICAgIGlmIChhcmdzID09IG51bGwpIHtcbiAgICAgIGFyZ3MgPSB7fTtcbiAgICB9XG4gICAgc3VwZXIoYXJncyk7XG5cbiAgICB0aGlzLnN1cHBvcnRzTWFza2luZyA9IHRydWU7XG4gICAgdGhpcy5heGlzID0gYXJncy5heGlzID09IG51bGwgPyAtMSA6IGFyZ3MuYXhpcztcbiAgICB0aGlzLm1vbWVudHVtID0gYXJncy5tb21lbnR1bSA9PSBudWxsID8gMC45OSA6IGFyZ3MubW9tZW50dW07XG4gICAgdGhpcy5lcHNpbG9uID0gYXJncy5lcHNpbG9uID09IG51bGwgPyAxZS0zIDogYXJncy5lcHNpbG9uO1xuICAgIHRoaXMuY2VudGVyID0gYXJncy5jZW50ZXIgPT0gbnVsbCA/IHRydWUgOiBhcmdzLmNlbnRlcjtcbiAgICB0aGlzLnNjYWxlID0gYXJncy5zY2FsZSA9PSBudWxsID8gdHJ1ZSA6IGFyZ3Muc2NhbGU7XG4gICAgdGhpcy5iZXRhSW5pdGlhbGl6ZXIgPSBnZXRJbml0aWFsaXplcihhcmdzLmJldGFJbml0aWFsaXplciB8fCAnemVyb3MnKTtcbiAgICB0aGlzLmdhbW1hSW5pdGlhbGl6ZXIgPSBnZXRJbml0aWFsaXplcihhcmdzLmdhbW1hSW5pdGlhbGl6ZXIgfHwgJ29uZXMnKTtcbiAgICB0aGlzLm1vdmluZ01lYW5Jbml0aWFsaXplciA9XG4gICAgICAgIGdldEluaXRpYWxpemVyKGFyZ3MubW92aW5nTWVhbkluaXRpYWxpemVyIHx8ICd6ZXJvcycpO1xuICAgIHRoaXMubW92aW5nVmFyaWFuY2VJbml0aWFsaXplciA9XG4gICAgICAgIGdldEluaXRpYWxpemVyKGFyZ3MubW92aW5nVmFyaWFuY2VJbml0aWFsaXplciB8fCAnb25lcycpO1xuICAgIHRoaXMuYmV0YUNvbnN0cmFpbnQgPSBnZXRDb25zdHJhaW50KGFyZ3MuYmV0YUNvbnN0cmFpbnQpO1xuICAgIHRoaXMuZ2FtbWFDb25zdHJhaW50ID0gZ2V0Q29uc3RyYWludChhcmdzLmdhbW1hQ29uc3RyYWludCk7XG4gICAgdGhpcy5iZXRhUmVndWxhcml6ZXIgPSBnZXRSZWd1bGFyaXplcihhcmdzLmJldGFSZWd1bGFyaXplcik7XG4gICAgdGhpcy5nYW1tYVJlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoYXJncy5nYW1tYVJlZ3VsYXJpemVyKTtcbiAgfVxuXG4gIHB1YmxpYyBvdmVycmlkZSBidWlsZChpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogdm9pZCB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBheGlzID0gdGhpcy5heGlzID49IDAgPyB0aGlzLmF4aXMgOiAodGhpcy5heGlzICsgaW5wdXRTaGFwZS5sZW5ndGgpO1xuICAgIGNvbnN0IGRpbSA9IGlucHV0U2hhcGVbYXhpc107XG4gICAgaWYgKGRpbSA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgQXhpcyAke2F4aXN9IG9mIGlucHV0IHRlbnNvciBzaG91bGQgaGF2ZSBhIGRlZmluZWQgZGltZW5zaW9uIGJ1dCBgICtcbiAgICAgICAgICBgdGhlIGxheWVyIHJlY2VpdmVkIGFuIGlucHV0IHdpdGggc2hhcGUgYCArXG4gICAgICAgICAgYCR7SlNPTi5zdHJpbmdpZnkoaW5wdXRTaGFwZSl9LmApO1xuICAgIH1cbiAgICB0aGlzLmlucHV0U3BlYyA9XG4gICAgICAgIFtuZXcgSW5wdXRTcGVjKHtuZGltOiBpbnB1dFNoYXBlLmxlbmd0aCwgYXhlczoge1theGlzXTogZGltfX0pXTtcbiAgICBjb25zdCBzaGFwZSA9IFtkaW1dO1xuICAgIGlmICh0aGlzLnNjYWxlKSB7XG4gICAgICB0aGlzLmdhbW1hID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICAgJ2dhbW1hJywgc2hhcGUsIG51bGwsIHRoaXMuZ2FtbWFJbml0aWFsaXplciwgdGhpcy5nYW1tYVJlZ3VsYXJpemVyLFxuICAgICAgICAgIHRydWUsIHRoaXMuZ2FtbWFDb25zdHJhaW50KTtcbiAgICB9XG4gICAgaWYgKHRoaXMuY2VudGVyKSB7XG4gICAgICB0aGlzLmJldGEgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgICAnYmV0YScsIHNoYXBlLCBudWxsLCB0aGlzLmJldGFJbml0aWFsaXplciwgdGhpcy5iZXRhUmVndWxhcml6ZXIsIHRydWUsXG4gICAgICAgICAgdGhpcy5iZXRhQ29uc3RyYWludCk7XG4gICAgfVxuICAgIHRoaXMubW92aW5nTWVhbiA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAnbW92aW5nX21lYW4nLCBzaGFwZSwgbnVsbCwgdGhpcy5tb3ZpbmdNZWFuSW5pdGlhbGl6ZXIsIG51bGwsIGZhbHNlKTtcbiAgICB0aGlzLm1vdmluZ1ZhcmlhbmNlID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICdtb3ZpbmdfdmFyaWFuY2UnLCBzaGFwZSwgbnVsbCwgdGhpcy5tb3ZpbmdWYXJpYW5jZUluaXRpYWxpemVyLCBudWxsLFxuICAgICAgICBmYWxzZSk7XG4gICAgdGhpcy5idWlsdCA9IHRydWU7XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgdHJhaW5pbmcgPSBrd2FyZ3NbJ3RyYWluaW5nJ10gPT0gbnVsbCA/IGZhbHNlIDoga3dhcmdzWyd0cmFpbmluZyddO1xuICAgICAgY29uc3QgaW5wdXQgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XG4gICAgICBjb25zdCBpbnB1dFNoYXBlID0gaW5wdXQuc2hhcGU7XG4gICAgICBjb25zdCBuZGltID0gaW5wdXRTaGFwZS5sZW5ndGg7XG4gICAgICBjb25zdCByZWR1Y3Rpb25BeGVzID0gbWF0aF91dGlscy5yYW5nZSgwLCBuZGltKTtcbiAgICAgIGNvbnN0IGF4aXMgPSB0aGlzLmF4aXMgPj0gMCA/IHRoaXMuYXhpcyA6ICh0aGlzLmF4aXMgKyBuZGltKTtcbiAgICAgIHJlZHVjdGlvbkF4ZXMuc3BsaWNlKGF4aXMsIDEpO1xuICAgICAgY29uc3QgYnJvYWRjYXN0U2hhcGUgPSBnZW5lcmljX3V0aWxzLnB5TGlzdFJlcGVhdCgxLCBuZGltKTtcbiAgICAgIGJyb2FkY2FzdFNoYXBlW2F4aXNdID0gaW5wdXRTaGFwZVtheGlzXTtcblxuICAgICAgY29uc3Qgc29ydGVkUmVkdWN0aW9uQXhlcyA9IHJlZHVjdGlvbkF4ZXMuc2xpY2UoKTtcbiAgICAgIHNvcnRlZFJlZHVjdGlvbkF4ZXMuc29ydCgpO1xuICAgICAgY29uc3QgbmVlZHNCcm9hZGNhc3RpbmcgPSAhdXRpbC5hcnJheXNFcXVhbChcbiAgICAgICAgICBzb3J0ZWRSZWR1Y3Rpb25BeGVzLCBtYXRoX3V0aWxzLnJhbmdlKDAsIG5kaW0pLnNsaWNlKDAsIG5kaW0gLSAxKSk7XG5cbiAgICAgIGNvbnN0IG5vcm1hbGl6ZUluZmVyZW5jZTogKCkgPT4gVGVuc29yID0gKCkgPT4ge1xuICAgICAgICBpZiAobmVlZHNCcm9hZGNhc3RpbmcpIHtcbiAgICAgICAgICBjb25zdCBicm9hZGNhc3RNb3ZpbmdNZWFuID1cbiAgICAgICAgICAgICAgcmVzaGFwZSh0aGlzLm1vdmluZ01lYW4ucmVhZCgpLCBicm9hZGNhc3RTaGFwZSk7XG4gICAgICAgICAgY29uc3QgYnJvYWRjYXN0TW92aW5nVmFyaWFuY2UgPVxuICAgICAgICAgICAgICByZXNoYXBlKHRoaXMubW92aW5nVmFyaWFuY2UucmVhZCgpLCBicm9hZGNhc3RTaGFwZSk7XG4gICAgICAgICAgY29uc3QgYnJvYWRjYXN0QmV0YSA9XG4gICAgICAgICAgICAgIHRoaXMuY2VudGVyID8gcmVzaGFwZSh0aGlzLmJldGEucmVhZCgpLCBicm9hZGNhc3RTaGFwZSkgOiBudWxsO1xuICAgICAgICAgIGNvbnN0IGJyb2FkY2FzdEdhbW1hID1cbiAgICAgICAgICAgICAgdGhpcy5zY2FsZSA/IHJlc2hhcGUodGhpcy5nYW1tYS5yZWFkKCksIGJyb2FkY2FzdFNoYXBlKSA6IG51bGw7XG4gICAgICAgICAgcmV0dXJuIGJhdGNoTm9ybWFsaXphdGlvbihcbiAgICAgICAgICAgICAgaW5wdXQsIGJyb2FkY2FzdE1vdmluZ01lYW4sIGJyb2FkY2FzdE1vdmluZ1ZhcmlhbmNlLFxuICAgICAgICAgICAgICBicm9hZGNhc3RCZXRhLCBicm9hZGNhc3RHYW1tYSwgdGhpcy5lcHNpbG9uKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICByZXR1cm4gYmF0Y2hOb3JtYWxpemF0aW9uKFxuICAgICAgICAgICAgICBpbnB1dCwgdGhpcy5tb3ZpbmdNZWFuLnJlYWQoKSwgdGhpcy5tb3ZpbmdWYXJpYW5jZS5yZWFkKCksXG4gICAgICAgICAgICAgIHRoaXMuYmV0YSA9PSBudWxsID8gbnVsbCA6IHRoaXMuYmV0YS5yZWFkKCksXG4gICAgICAgICAgICAgIHRoaXMuZ2FtbWEgPT0gbnVsbCA/IG51bGwgOiB0aGlzLmdhbW1hLnJlYWQoKSwgdGhpcy5lcHNpbG9uKTtcbiAgICAgICAgfVxuICAgICAgfTtcblxuICAgICAgaWYgKCF0cmFpbmluZykge1xuICAgICAgICByZXR1cm4gbm9ybWFsaXplSW5mZXJlbmNlKCk7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IFtub3JtZWRUcmFpbmluZywgbWVhbiwgdmFyaWFuY2VdID0gbm9ybWFsaXplQmF0Y2hJblRyYWluaW5nKFxuICAgICAgICAgIGlucHV0LCB0aGlzLmdhbW1hLnJlYWQoKSwgdGhpcy5iZXRhLnJlYWQoKSwgcmVkdWN0aW9uQXhlcyxcbiAgICAgICAgICB0aGlzLmVwc2lsb24pO1xuXG4gICAgICBjb25zdCBkb01vdmluZ0F2ZXJhZ2UgPVxuICAgICAgICAgICh2YXJpYWJsZTogTGF5ZXJWYXJpYWJsZSwgdmFsdWU6IFRlbnNvciwgbW9tZW50dW06IG51bWJlcik6IHZvaWQgPT4ge1xuICAgICAgICAgICAgdGZjLnRpZHkoKCkgPT4ge1xuICAgICAgICAgICAgICBjb25zdCBkZWNheSA9IDEgLSBtb21lbnR1bTtcbiAgICAgICAgICAgICAgY29uc3Qgb3JpZ1ZhbHVlID0gdmFyaWFibGUucmVhZCgpO1xuICAgICAgICAgICAgICBjb25zdCB1cGRhdGVEZWx0YSA9IHRmYy5tdWwodGZjLnN1YihvcmlnVmFsdWUsIHZhbHVlKSwgZGVjYXkpO1xuICAgICAgICAgICAgICB2YXJpYWJsZS53cml0ZSh0ZmMuc3ViKG9yaWdWYWx1ZSwgdXBkYXRlRGVsdGEpKTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgIH07XG5cbiAgICAgIC8vIFBlcmZvcm0gdXBkYXRlcyB0byBtb3ZpbmcgbWVhbiBhbmQgbW92aW5nIHZhcmlhbmNlIGZvciB0cmFpbmluZy5cbiAgICAgIC8vIFBvcnRpbmcgTm90ZTogSW4gUHlLZXJhcywgdGhlc2UgdXBkYXRlcyB0byBgbW92aW5nTWVhbmAgYW5kXG4gICAgICAvLyAgIGBtb3ZpbmdBdmVyYWdlYCBhcmUgZG9uZSBhcyBhIGRlZmVycmVkIEdyYXBoLCBhZGRlZCB0byB0aGUgYExheWVyYCdzXG4gICAgICAvLyAgIGB1cGRhdGVgcyB1c2luZyB0aGUgYGFkZF91cGRhdGUoKWAgbWV0aG9kLiBIZXJlIHdlIGRvIGl0IGltcGVyYXRpdmVseVxuICAgICAgLy8gICBhbmQgZW5jYXBzdWxhdGUgdGhlIHVwZGF0ZXMgaW4gYSBmdW5jdGlvbiB0aGF0IGlzIGludm9rZWRcbiAgICAgIC8vICAgaW1tZWRpYXRlbHkuXG4gICAgICBjb25zdCB1cGRhdGVNb3ZpbmdNZWFuQW5kVmFyaWFuY2UgPSAoKSA9PiB7XG4gICAgICAgIGRvTW92aW5nQXZlcmFnZSh0aGlzLm1vdmluZ01lYW4sIG1lYW4sIHRoaXMubW9tZW50dW0pO1xuICAgICAgICBkb01vdmluZ0F2ZXJhZ2UodGhpcy5tb3ZpbmdWYXJpYW5jZSwgdmFyaWFuY2UsIHRoaXMubW9tZW50dW0pO1xuICAgICAgfTtcbiAgICAgIHVwZGF0ZU1vdmluZ01lYW5BbmRWYXJpYW5jZSgpO1xuXG4gICAgICByZXR1cm4gbm9ybWVkVHJhaW5pbmc7XG4gICAgfSk7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtcbiAgICAgIGF4aXM6IHRoaXMuYXhpcyxcbiAgICAgIG1vbWVudHVtOiB0aGlzLm1vbWVudHVtLFxuICAgICAgZXBzaWxvbjogdGhpcy5lcHNpbG9uLFxuICAgICAgY2VudGVyOiB0aGlzLmNlbnRlcixcbiAgICAgIHNjYWxlOiB0aGlzLnNjYWxlLFxuICAgICAgYmV0YUluaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLmJldGFJbml0aWFsaXplciksXG4gICAgICBnYW1tYUluaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLmdhbW1hSW5pdGlhbGl6ZXIpLFxuICAgICAgbW92aW5nTWVhbkluaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLm1vdmluZ01lYW5Jbml0aWFsaXplciksXG4gICAgICBtb3ZpbmdWYXJpYW5jZUluaXRpYWxpemVyOlxuICAgICAgICAgIHNlcmlhbGl6ZUluaXRpYWxpemVyKHRoaXMubW92aW5nVmFyaWFuY2VJbml0aWFsaXplciksXG4gICAgICBiZXRhUmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMuYmV0YVJlZ3VsYXJpemVyKSxcbiAgICAgIGdhbW1hUmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMuZ2FtbWFSZWd1bGFyaXplciksXG4gICAgICBiZXRhQ29uc3RyYWludDogc2VyaWFsaXplQ29uc3RyYWludCh0aGlzLmJldGFDb25zdHJhaW50KSxcbiAgICAgIGdhbW1hQ29uc3RyYWludDogc2VyaWFsaXplQ29uc3RyYWludCh0aGlzLmdhbW1hQ29uc3RyYWludClcbiAgICB9O1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBPYmplY3QuYXNzaWduKGNvbmZpZywgYmFzZUNvbmZpZyk7XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEJhdGNoTm9ybWFsaXphdGlvbik7XG5cbmV4cG9ydCBpbnRlcmZhY2UgTGF5ZXJOb3JtYWxpemF0aW9uTGF5ZXJBcmdzIGV4dGVuZHMgTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIFRoZSBheGlzIG9yIGF4ZXMgdGhhdCBzaG91bGQgYmUgbm9ybWFsaXplZCAodHlwaWNhbGx5LCB0aGUgZmVhdHVyZSBheGlzKS5cbiAgICogRGVmYXVsdHMgdG8gLTEgKHRoZSBsYXN0IGF4aXMpLlxuICAgKi9cbiAgYXhpcz86IG51bWJlcnxudW1iZXJbXTtcblxuICAvKipcbiAgICogQSBzbWFsbCBwb3NpdGl2ZSBmbG9hdCBhZGRlZCB0byB2YXJpYW5jZSB0byBhdm9pZCBkaXZpc29uIGJ5IHplcm8uXG4gICAqIERlZmF1bHRzIHRvIDFlLTMuXG4gICAqL1xuICBlcHNpbG9uPzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBJZiBgdHJ1ZWAsIGFkZCBvZmZzZXQgb2YgYGJldGFgIHRvIG5vcm1hbGl6ZWQgdGVuc29yLlxuICAgKiBJZiBgZmFsc2VgLCBgYmV0YWAgaXMgaWdub3JlZC5cbiAgICogRGVmYXVsdDogYHRydWVgLlxuICAgKi9cbiAgY2VudGVyPzogYm9vbGVhbjtcblxuICAvKipcbiAgICogSWYgYHRydWVgLCBtdWx0aXBseSBvdXRwdXQgYnkgYGdhbW1hYC5cbiAgICogSWYgYGZhbHNlYCwgYGdhbW1hYCBpcyBub3QgdXNlZC5cbiAgICogV2hlbiB0aGUgbmV4dCBsYXllciBpcyBsaW5lYXIsIHRoaXMgY2FuIGJlIGRpc2FibGVkIHNpbmNlIHNjYWxpbmcgd2lsbFxuICAgKiBiZSBkb25lIGJ5IHRoZSBuZXh0IGxheWVyLlxuICAgKiBEZWZhdWx0OiBgdHJ1ZWAuXG4gICAqL1xuICBzY2FsZT86IGJvb2xlYW47XG5cbiAgLyoqXG4gICAqIEluaXRpYWxpemVyIGZvciB0aGUgYmV0YSB3ZWlnaHQuXG4gICAqIERlZmF1bHQ6IGAnemVyb3MnYC5cbiAgICovXG4gIGJldGFJbml0aWFsaXplcj86IEluaXRpYWxpemVySWRlbnRpZmllcnxJbml0aWFsaXplcjtcblxuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBnYW1tYSB3ZWlnaHQuXG4gICAqIERlZmF1bHQ6IGAnb25lcydgLlxuICAgKi9cbiAgZ2FtbWFJbml0aWFsaXplcj86IEluaXRpYWxpemVySWRlbnRpZmllcnxJbml0aWFsaXplcjtcblxuICAvKiogUmVndWxhcml6ZXIgZm9yIHRoZSBiZXRhIHdlaWdodC4gKi9cbiAgYmV0YVJlZ3VsYXJpemVyPzogUmVndWxhcml6ZXJJZGVudGlmaWVyfFJlZ3VsYXJpemVyO1xuXG4gIC8qKiBSZWd1bGFyaXplciBmb3IgdGhlIGdhbW1hIHdlaWdodC4gKi9cbiAgZ2FtbWFSZWd1bGFyaXplcj86IFJlZ3VsYXJpemVySWRlbnRpZmllcnxSZWd1bGFyaXplcjtcbn1cblxuZXhwb3J0IGNsYXNzIExheWVyTm9ybWFsaXphdGlvbiBleHRlbmRzIExheWVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnTGF5ZXJOb3JtYWxpemF0aW9uJztcblxuICBwcml2YXRlIGF4aXM6IG51bWJlcnxudW1iZXJbXTtcbiAgcmVhZG9ubHkgZXBzaWxvbjogbnVtYmVyO1xuICByZWFkb25seSBjZW50ZXI6IGJvb2xlYW47XG4gIHJlYWRvbmx5IHNjYWxlOiBib29sZWFuO1xuICByZWFkb25seSBiZXRhSW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICByZWFkb25seSBnYW1tYUluaXRpYWxpemVyOiBJbml0aWFsaXplcjtcbiAgcmVhZG9ubHkgYmV0YVJlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcbiAgcmVhZG9ubHkgZ2FtbWFSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG5cbiAgcHJpdmF0ZSBnYW1tYTogTGF5ZXJWYXJpYWJsZTtcbiAgcHJpdmF0ZSBiZXRhOiBMYXllclZhcmlhYmxlO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBMYXllck5vcm1hbGl6YXRpb25MYXllckFyZ3MpIHtcbiAgICBpZiAoYXJncyA9PSBudWxsKSB7XG4gICAgICBhcmdzID0ge307XG4gICAgfVxuICAgIHN1cGVyKGFyZ3MpO1xuXG4gICAgdGhpcy5heGlzID0gYXJncy5heGlzID09IG51bGwgPyAtMSA6IGFyZ3MuYXhpcztcbiAgICBpZiAodHlwZW9mIHRoaXMuYXhpcyA9PT0gJ251bWJlcicpIHtcbiAgICAgIGlmICghTnVtYmVyLmlzSW50ZWdlcih0aGlzLmF4aXMpKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgIGBFeHBlY3RlZCBheGlzIHRvIGJlIGFuIGludGVnZXIsIGJ1dCByZWNlaXZlZCAke3RoaXMuYXhpc31gKTtcbiAgICAgIH1cbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5heGlzKSkge1xuICAgICAgZm9yIChjb25zdCBheGlzIG9mIHRoaXMuYXhpcykge1xuICAgICAgICBpZiAoIU51bWJlci5pc0ludGVnZXIoYXhpcykpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAgIGBFeHBlY3RlZCBheGlzIHRvIGJlIGFuIGFycmF5IG9mIGludGVnZXJzLCBgICtcbiAgICAgICAgICAgICAgYGJ1dCByZWNlaXZlZCAke0pTT04uc3RyaW5naWZ5KHRoaXMuYXhpcyl9YCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBFeHBlY3RlZCBheGlzIHRvIGJlIGFuIGludGVnZXIgb3IgYW4gYXJyYXkgb2YgaW50ZWdlcnMsIGAgK1xuICAgICAgICAgIGBidXQgcmVjZWl2ZWQgJHtKU09OLnN0cmluZ2lmeSh0aGlzLmF4aXMpfWApO1xuICAgIH1cblxuICAgIHRoaXMuZXBzaWxvbiA9IGFyZ3MuZXBzaWxvbiA9PSBudWxsID8gMWUtMyA6IGFyZ3MuZXBzaWxvbjtcbiAgICB0aGlzLmNlbnRlciA9IGFyZ3MuY2VudGVyID09IG51bGwgPyB0cnVlIDogYXJncy5jZW50ZXI7XG4gICAgdGhpcy5zY2FsZSA9IGFyZ3Muc2NhbGUgPT0gbnVsbCA/IHRydWUgOiBhcmdzLnNjYWxlO1xuICAgIHRoaXMuYmV0YUluaXRpYWxpemVyID0gZ2V0SW5pdGlhbGl6ZXIoYXJncy5iZXRhSW5pdGlhbGl6ZXIgfHwgJ3plcm9zJyk7XG4gICAgdGhpcy5nYW1tYUluaXRpYWxpemVyID0gZ2V0SW5pdGlhbGl6ZXIoYXJncy5nYW1tYUluaXRpYWxpemVyIHx8ICdvbmVzJyk7XG4gICAgdGhpcy5iZXRhUmVndWxhcml6ZXIgPSBnZXRSZWd1bGFyaXplcihhcmdzLmJldGFSZWd1bGFyaXplcik7XG4gICAgdGhpcy5nYW1tYVJlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoYXJncy5nYW1tYVJlZ3VsYXJpemVyKTtcblxuICAgIHRoaXMuc3VwcG9ydHNNYXNraW5nID0gdHJ1ZTtcbiAgfVxuXG4gIHB1YmxpYyBvdmVycmlkZSBidWlsZChpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogdm9pZCB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBuRGltcyA9IGlucHV0U2hhcGUubGVuZ3RoO1xuXG4gICAgLy8gQ29udmVydCBheGlzIHRvIGFycmF5IGFuZCByZXNvbHZlIG5lZ2F0aXZlcy5cbiAgICBpZiAodHlwZW9mIHRoaXMuYXhpcyA9PT0gJ251bWJlcicpIHtcbiAgICAgIHRoaXMuYXhpcyA9IFt0aGlzLmF4aXNdO1xuICAgIH1cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuYXhpcy5sZW5ndGg7ICsraSkge1xuICAgICAgaWYgKHRoaXMuYXhpc1tpXSA8IDApIHtcbiAgICAgICAgdGhpcy5heGlzW2ldICs9IG5EaW1zO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIEZ1cnRoZXIgdmFsaWRhdGUgYXhlcy5cbiAgICBmb3IgKGNvbnN0IGF4aXMgb2YgdGhpcy5heGlzKSB7XG4gICAgICBpZiAoYXhpcyA8IDAgfHwgYXhpcyA+PSBuRGltcykge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoYEludmFsaWQgYXhpczogJHtheGlzfWApO1xuICAgICAgfVxuICAgIH1cbiAgICBpZiAodGhpcy5heGlzLmxlbmd0aCAhPT0gZ2VuZXJpY191dGlscy51bmlxdWUodGhpcy5heGlzKS5sZW5ndGgpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgRm91bmQgZHVwbGljYXRlIGF4ZXMgaW46ICR7dGhpcy5heGlzfWApO1xuICAgIH1cblxuICAgIGNvbnN0IHBhcmFtU2hhcGUgPSB0aGlzLmF4aXMubWFwKGF4aXMgPT4gaW5wdXRTaGFwZVtheGlzXSkgYXMgbnVtYmVyW107XG5cbiAgICBjb25zdCB0cmFpbmFibGUgPSB0cnVlO1xuICAgIGlmICh0aGlzLnNjYWxlKSB7XG4gICAgICB0aGlzLmdhbW1hID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICAgJ2dhbW1hJywgcGFyYW1TaGFwZSwgJ2Zsb2F0MzInLCB0aGlzLmdhbW1hSW5pdGlhbGl6ZXIsXG4gICAgICAgICAgdGhpcy5nYW1tYVJlZ3VsYXJpemVyLCB0cmFpbmFibGUpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmdhbW1hID0gbnVsbDtcbiAgICB9XG4gICAgaWYgKHRoaXMuY2VudGVyKSB7XG4gICAgICB0aGlzLmJldGEgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgICAnYmV0YScsIHBhcmFtU2hhcGUsICdmbG9hdDMyJywgdGhpcy5iZXRhSW5pdGlhbGl6ZXIsXG4gICAgICAgICAgdGhpcy5iZXRhUmVndWxhcml6ZXIsIHRyYWluYWJsZSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuYmV0YSA9IG51bGw7XG4gICAgfVxuXG4gICAgdGhpcy5idWlsdCA9IHRydWU7XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgY29uc3QgaW5wdXQgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XG4gICAgY29uc3QgaW5wdXRTaGFwZSA9IGlucHV0LnNoYXBlO1xuICAgIGNvbnN0IG5EaW1zID0gaW5wdXRTaGFwZS5sZW5ndGg7XG5cbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBrZWVwRGltcyA9IHRydWU7XG4gICAgICBsZXQge21lYW4sIHZhcmlhbmNlfSA9IG1vbWVudHMoaW5wdXQsIHRoaXMuYXhpcywga2VlcERpbXMpO1xuICAgICAgY29uc3QgYnJvYWRjYXN0U2hhcGUgPSBnZW5lcmljX3V0aWxzLnB5TGlzdFJlcGVhdCgxLCBuRGltcyk7XG4gICAgICBmb3IgKGNvbnN0IGRpbSBvZiB0aGlzLmF4aXMgYXMgbnVtYmVyW10pIHtcbiAgICAgICAgYnJvYWRjYXN0U2hhcGVbZGltXSA9IGlucHV0U2hhcGVbZGltXTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgYnJvYWRjYXN0ID0gKHY6IFRlbnNvcikgPT4ge1xuICAgICAgICBpZiAodiAhPSBudWxsICYmIHYuc2hhcGUubGVuZ3RoICE9PSBuRGltcykge1xuICAgICAgICAgIHJldHVybiB0ZmMucmVzaGFwZSh2LCBicm9hZGNhc3RTaGFwZSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgcmV0dXJuIHY7XG4gICAgICAgIH1cbiAgICAgIH07XG5cbiAgICAgIGxldCBzY2FsZSA9IHRoaXMuc2NhbGUgPyBicm9hZGNhc3QodGhpcy5nYW1tYS5yZWFkKCkpIDogbnVsbDtcbiAgICAgIGxldCBvZmZzZXQgPSB0aGlzLmNlbnRlciA/IGJyb2FkY2FzdCh0aGlzLmJldGEucmVhZCgpKSA6IG51bGw7XG5cbiAgICAgIC8vIFRPRE8oaHR0cHM6Ly9naXRodWIuY29tL3RlbnNvcmZsb3cvdGZqcy9pc3N1ZXMvMjEyMCk6IFRoZSB0aWxpbmcgYmVsb3dcbiAgICAgIC8vIGlzIGEgd29ya2Fyb3VuZCBmb3IgdGhlIGxpbWl0YXRpb24gb2YgY29yZSdzIGJhdGNoTm9ybWFsaXphdGlvbj9kIGRvbid0XG4gICAgICAvLyBzdXBwb3J0IGJyb2FkY2FzdGluZyBpbiB0aGVpciBncmFkaWVudHMuIEluIGFkZGl0aW9uLCB0aGUgdGlsaW5nIGlzXG4gICAgICAvLyBuZWNlc3NhcnkgdG8gZW5zdXJlIGNvcnJlY3RuZXNzIG9uIHRoZSBicm93c2VyIENQVSBiYWNrZW5kIHJlZ2FyZGxlc3NcbiAgICAgIC8vIG9mIGZvcndhcmQgb3IgYmFja3dhcmQgY29tcHV0YXRpb24uIFJlbW92ZSB0aGlzIHdvcmthcm91bmQgb25jZSB0aGVcbiAgICAgIC8vIGxpbWl0YXRpb24gaXMgYWRkcmVzc2VkLiBTZWUgLlxuICAgICAgY29uc3QgbW9tZW50c1RpbGluZzogbnVtYmVyW10gPSBbXTtcbiAgICAgIGNvbnN0IHNjYWxlT2Zmc2V0VGlsaW5nOiBudW1iZXJbXSA9IFtdO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBuRGltczsgKytpKSB7XG4gICAgICAgIGlmICgodGhpcy5heGlzIGFzIG51bWJlcltdKS5pbmRleE9mKGkpICE9PSAtMSkge1xuICAgICAgICAgIG1vbWVudHNUaWxpbmcucHVzaChpbnB1dFNoYXBlW2ldKTtcbiAgICAgICAgICBzY2FsZU9mZnNldFRpbGluZy5wdXNoKDEpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIG1vbWVudHNUaWxpbmcucHVzaCgxKTtcbiAgICAgICAgICBzY2FsZU9mZnNldFRpbGluZy5wdXNoKGlucHV0U2hhcGVbaV0pO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBtZWFuID0gdGZjLnRpbGUobWVhbiwgbW9tZW50c1RpbGluZyk7XG4gICAgICB2YXJpYW5jZSA9IHRmYy50aWxlKHZhcmlhbmNlLCBtb21lbnRzVGlsaW5nKTtcbiAgICAgIGlmIChzY2FsZSAhPSBudWxsKSB7XG4gICAgICAgIHNjYWxlID0gdGZjLnRpbGUoc2NhbGUsIHNjYWxlT2Zmc2V0VGlsaW5nKTtcbiAgICAgIH1cbiAgICAgIGlmIChvZmZzZXQgIT0gbnVsbCkge1xuICAgICAgICBvZmZzZXQgPSB0ZmMudGlsZShvZmZzZXQsIHNjYWxlT2Zmc2V0VGlsaW5nKTtcbiAgICAgIH1cblxuICAgICAgcmV0dXJuIGJhdGNoTm9ybWFsaXphdGlvbihcbiAgICAgICAgICBpbnB1dCwgbWVhbiwgdmFyaWFuY2UsIG9mZnNldCwgc2NhbGUsIHRoaXMuZXBzaWxvbik7XG4gICAgfSk7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtcbiAgICAgIGF4aXM6IHRoaXMuYXhpcyxcbiAgICAgIGVwc2lsb246IHRoaXMuZXBzaWxvbixcbiAgICAgIGNlbnRlcjogdGhpcy5jZW50ZXIsXG4gICAgICBzY2FsZTogdGhpcy5zY2FsZSxcbiAgICAgIGJldGFJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5iZXRhSW5pdGlhbGl6ZXIpLFxuICAgICAgZ2FtbWFJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5nYW1tYUluaXRpYWxpemVyKSxcbiAgICAgIGJldGFSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5iZXRhUmVndWxhcml6ZXIpLFxuICAgICAgZ2FtbWFSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5nYW1tYVJlZ3VsYXJpemVyKVxuICAgIH07XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoTGF5ZXJOb3JtYWxpemF0aW9uKTtcbiJdfQ==