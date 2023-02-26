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
 * Layers that augment the functionality of a base layer.
 */
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy } from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import { nameScope } from '../common';
import { InputSpec, Layer, SymbolicTensor } from '../engine/topology';
import { NotImplementedError, ValueError } from '../errors';
import { VALID_BIDIRECTIONAL_MERGE_MODES } from '../keras_format/common';
import * as generic_utils from '../utils/generic_utils';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';
import { rnn, standardizeArgs } from './recurrent';
import { deserialize } from './serialization';
/**
 * Abstract wrapper base class.
 *
 * Wrappers take another layer and augment it in various ways.
 * Do not use this class as a layer, it is only an abstract base class.
 * Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.
 */
export class Wrapper extends Layer {
    constructor(args) {
        // Porting Note: In PyKeras, `self.layer` is set prior to the calling
        //   `super()`. But we can't do that here due to TypeScript's restriction.
        //   See: https://github.com/Microsoft/TypeScript/issues/8277
        //   As a result, we have to add checks in `get trainable()` and
        //   `set trainable()` below in order to prevent using `this.layer` when
        //   its value is `undefined`. The super constructor does use the getter
        //   and the setter of `this.layer`.
        super(args);
        this.layer = args.layer;
    }
    build(inputShape) {
        this.built = true;
    }
    // TODO(cais): Implement activityRegularizer getter.
    get trainable() {
        // Porting Note: the check of `this.layer` here is necessary due to the
        //   way the `constructor` of this class is written (see Porting Note
        //   above).
        if (this.layer != null) {
            return this.layer.trainable;
        }
        else {
            return false;
        }
    }
    set trainable(value) {
        // Porting Note: the check of `this.layer` here is necessary due to the
        //   way the `constructor` of this class is written (see Porting Note
        //   above).
        if (this.layer != null) {
            this.layer.trainable = value;
        }
    }
    get trainableWeights() {
        return this.layer.trainableWeights;
    }
    // TODO(cais): Implement setter for trainableWeights.
    get nonTrainableWeights() {
        return this.layer.nonTrainableWeights;
    }
    // TODO(cais): Implement setter for nonTrainableWeights.
    get updates() {
        // tslint:disable-next-line:no-any
        return this.layer._updates;
    }
    // TODO(cais): Implement getUpdatesFor().
    get losses() {
        return this.layer.losses;
    }
    // TODO(cais): Implement getLossesFor().
    getWeights() {
        return this.layer.getWeights();
    }
    setWeights(weights) {
        this.layer.setWeights(weights);
    }
    getConfig() {
        const config = {
            'layer': {
                'className': this.layer.getClassName(),
                'config': this.layer.getConfig(),
            }
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    setFastWeightInitDuringBuild(value) {
        super.setFastWeightInitDuringBuild(value);
        if (this.layer != null) {
            this.layer.setFastWeightInitDuringBuild(value);
        }
    }
    /** @nocollapse */
    static fromConfig(cls, config, customObjects = {}) {
        const layerConfig = config['layer'];
        const layer = deserialize(layerConfig, customObjects);
        delete config['layer'];
        const newConfig = { layer };
        Object.assign(newConfig, config);
        return new cls(newConfig);
    }
}
export class TimeDistributed extends Wrapper {
    constructor(args) {
        super(args);
        this.supportsMasking = true;
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        if (inputShape.length < 3) {
            throw new ValueError(`TimeDistributed layer expects an input shape >= 3D, but received ` +
                `input shape ${JSON.stringify(inputShape)}`);
        }
        this.inputSpec = [{ shape: inputShape }];
        const childInputShape = [inputShape[0]].concat(inputShape.slice(2));
        if (!this.layer.built) {
            this.layer.build(childInputShape);
            this.layer.built = true;
        }
        super.build(inputShape);
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const childInputShape = [inputShape[0]].concat(inputShape.slice(2));
        const childOutputShape = this.layer.computeOutputShape(childInputShape);
        const timesteps = inputShape[1];
        return [childOutputShape[0], timesteps].concat(childOutputShape.slice(1));
    }
    call(inputs, kwargs) {
        return tidy(() => {
            // TODO(cais): Add 'training' and 'useLearningPhase' to kwargs.
            inputs = getExactlyOneTensor(inputs);
            // Porting Note: In tfjs-layers, `inputs` are always concrete tensor
            // values. Hence the inputs can't have an undetermined first (batch)
            // dimension, which is why we always use the K.rnn approach here.
            const step = (inputs, states) => {
                // TODO(cais): Add useLearningPhase.
                // NOTE(cais): `layer.call` may return a length-1 array of Tensor in
                //   some cases (e.g., `layer` is a `Sequential` instance), which is
                //   why `getExactlyOneTensor` is used below.
                const output = getExactlyOneTensor(this.layer.call(inputs, kwargs));
                return [output, []];
            };
            const rnnOutputs = rnn(step, inputs, [], false /* goBackwards */, null /* mask */, null /* constants */, false /* unroll */, true /* needPerStepOutputs */);
            const y = rnnOutputs[1];
            // TODO(cais): Add activity regularization.
            // TODO(cais): Add useLearningPhase.
            return y;
        });
    }
}
/** @nocollapse */
TimeDistributed.className = 'TimeDistributed';
serialization.registerClass(TimeDistributed);
export function checkBidirectionalMergeMode(value) {
    generic_utils.checkStringTypeUnionValue(VALID_BIDIRECTIONAL_MERGE_MODES, 'BidirectionalMergeMode', value);
}
const DEFAULT_BIDIRECTIONAL_MERGE_MODE = 'concat';
export class Bidirectional extends Wrapper {
    constructor(args) {
        super(args);
        // Note: When creating `this.forwardLayer`, the original Layer object
        //   (`config.layer`) ought to be cloned. This is why we call
        //   `getConfig()` followed by `deserialize()`. Without this cloning,
        //   the layer names saved during serialization will incorrectly contain
        //   the 'forward_' prefix. In Python Keras, this is done using
        //   `copy.copy` (shallow copy), which does not have a simple equivalent
        //   in JavaScript. JavaScript's `Object.assign()` does not copy
        //   methods.
        const layerConfig = args.layer.getConfig();
        const forwDict = {};
        forwDict['className'] = args.layer.getClassName();
        forwDict['config'] = layerConfig;
        this.forwardLayer = deserialize(forwDict);
        layerConfig['goBackwards'] =
            layerConfig['goBackwards'] === true ? false : true;
        const backDict = {};
        backDict['className'] = args.layer.getClassName();
        backDict['config'] = layerConfig;
        this.backwardLayer = deserialize(backDict);
        this.forwardLayer.name = 'forward_' + this.forwardLayer.name;
        this.backwardLayer.name = 'backward_' + this.backwardLayer.name;
        this.mergeMode = args.mergeMode === undefined ?
            DEFAULT_BIDIRECTIONAL_MERGE_MODE :
            args.mergeMode;
        checkBidirectionalMergeMode(this.mergeMode);
        if (args.weights) {
            throw new NotImplementedError('weights support is not implemented for Bidirectional layer yet.');
        }
        this._stateful = args.layer.stateful;
        this.returnSequences = args.layer.returnSequences;
        this.returnState = args.layer.returnState;
        this.supportsMasking = true;
        this._trainable = true;
        this.inputSpec = args.layer.inputSpec;
        this.numConstants = null;
    }
    get trainable() {
        return this._trainable;
    }
    set trainable(value) {
        // Porting Note: the check of `this.layer` here is necessary due to the
        //   way the `constructor` of this class is written (see Porting Note
        //   above).
        this._trainable = value;
        if (this.forwardLayer != null) {
            this.forwardLayer.trainable = value;
        }
        if (this.backwardLayer != null) {
            this.backwardLayer.trainable = value;
        }
    }
    getWeights() {
        return this.forwardLayer.getWeights().concat(this.backwardLayer.getWeights());
    }
    setWeights(weights) {
        const numWeights = weights.length;
        const numeightsOver2 = Math.floor(numWeights / 2);
        this.forwardLayer.setWeights(weights.slice(0, numeightsOver2));
        this.backwardLayer.setWeights(weights.slice(numeightsOver2));
    }
    computeOutputShape(inputShape) {
        let layerShapes = this.forwardLayer.computeOutputShape(inputShape);
        if (!(Array.isArray(layerShapes) && Array.isArray(layerShapes[0]))) {
            layerShapes = [layerShapes];
        }
        layerShapes = layerShapes;
        let outputShape;
        let outputShapes;
        let stateShape;
        if (this.returnState) {
            stateShape = layerShapes.slice(1);
            outputShape = layerShapes[0];
        }
        else {
            outputShape = layerShapes[0];
        }
        outputShape = outputShape;
        if (this.mergeMode === 'concat') {
            outputShape[outputShape.length - 1] *= 2;
            outputShapes = [outputShape];
        }
        else if (this.mergeMode == null) {
            outputShapes = [outputShape, outputShape.slice()];
        }
        else {
            outputShapes = [outputShape];
        }
        if (this.returnState) {
            if (this.mergeMode == null) {
                return outputShapes.concat(stateShape).concat(stateShape.slice());
            }
            return [outputShape].concat(stateShape).concat(stateShape.slice());
        }
        return generic_utils.singletonOrArray(outputShapes);
    }
    apply(inputs, kwargs) {
        let initialState = kwargs == null ? null : kwargs['initialState'];
        let constants = kwargs == null ? null : kwargs['constants'];
        if (kwargs == null) {
            kwargs = {};
        }
        const standardized = standardizeArgs(inputs, initialState, constants, this.numConstants);
        inputs = standardized.inputs;
        initialState = standardized.initialState;
        constants = standardized.constants;
        if (Array.isArray(inputs)) {
            initialState = inputs.slice(1);
            inputs = inputs[0];
        }
        if ((initialState == null || initialState.length === 0) &&
            constants == null) {
            return super.apply(inputs, kwargs);
        }
        const additionalInputs = [];
        const additionalSpecs = [];
        if (initialState != null) {
            const numStates = initialState.length;
            if (numStates % 2 > 0) {
                throw new ValueError('When passing `initialState` to a Bidrectional RNN, ' +
                    'the state should be an Array containing the states of ' +
                    'the underlying RNNs.');
            }
            kwargs['initialState'] = initialState;
            additionalInputs.push(...initialState);
            const stateSpecs = initialState
                .map(state => new InputSpec({ shape: state.shape }));
            this.forwardLayer.stateSpec = stateSpecs.slice(0, numStates / 2);
            this.backwardLayer.stateSpec = stateSpecs.slice(numStates / 2);
            additionalSpecs.push(...stateSpecs);
        }
        if (constants != null) {
            throw new NotImplementedError('Support for constants in Bidirectional layers is not ' +
                'implemented yet.');
        }
        const isSymbolicTensor = additionalInputs[0] instanceof SymbolicTensor;
        for (const tensor of additionalInputs) {
            if (tensor instanceof SymbolicTensor !== isSymbolicTensor) {
                throw new ValueError('The initial state of a Bidirectional layer cannot be ' +
                    'specified as a mix of symbolic and non-symbolic tensors');
            }
        }
        if (isSymbolicTensor) {
            // Compute the full input and specs, including the states.
            const fullInput = [inputs].concat(additionalInputs);
            const fullInputSpec = this.inputSpec.concat(additionalSpecs);
            // Perform the call temporarily and replace inputSpec.
            // Note: with initial states symbolic calls and non-symbolic calls to
            // this method differ in how the initial states are passed. For
            // symbolic calls, the initial states are passed in the first arg, as
            // an Array of SymbolicTensors; for non-symbolic calls, they are
            // passed in the second arg as a part of the kwargs. Hence the need to
            // temporarily modify inputSpec here.
            // TODO(cais): Make refactoring so that this hacky code below is no
            // longer needed.
            const originalInputSpec = this.inputSpec;
            this.inputSpec = fullInputSpec;
            const output = super.apply(fullInput, kwargs);
            this.inputSpec = originalInputSpec;
            return output;
        }
        else {
            return super.apply(inputs, kwargs);
        }
    }
    call(inputs, kwargs) {
        return tidy(() => {
            const initialState = kwargs['initialState'];
            let y;
            let yRev;
            if (initialState == null) {
                y = this.forwardLayer.call(inputs, kwargs);
                yRev = this.backwardLayer.call(inputs, kwargs);
            }
            else {
                const forwardState = initialState.slice(0, initialState.length / 2);
                const backwardState = initialState.slice(initialState.length / 2);
                y = this.forwardLayer.call(inputs, Object.assign(kwargs, { initialState: forwardState }));
                yRev = this.backwardLayer.call(inputs, Object.assign(kwargs, { initialState: backwardState }));
            }
            let states;
            if (this.returnState) {
                if (Array.isArray(y)) {
                    states = y.slice(1).concat(yRev.slice(1));
                }
                else {
                }
                y = y[0];
                yRev = yRev[0];
            }
            if (this.returnSequences) {
                yRev = tfc.reverse(yRev, 1);
            }
            let output;
            if (this.mergeMode === 'concat') {
                output = K.concatenate([y, yRev]);
            }
            else if (this.mergeMode === 'sum') {
                output = tfc.add(y, yRev);
            }
            else if (this.mergeMode === 'ave') {
                output = tfc.mul(.5, tfc.add(y, yRev));
            }
            else if (this.mergeMode === 'mul') {
                output = tfc.mul(y, yRev);
            }
            else if (this.mergeMode == null) {
                output = [y, yRev];
            }
            // TODO(cais): Properly set learning phase.
            if (this.returnState) {
                if (this.mergeMode == null) {
                    return output.concat(states);
                }
                return [output].concat(states);
            }
            return output;
        });
    }
    resetStates(states) {
        this.forwardLayer.resetStates();
        this.backwardLayer.resetStates();
    }
    build(inputShape) {
        nameScope(this.forwardLayer.name, () => {
            this.forwardLayer.build(inputShape);
        });
        nameScope(this.backwardLayer.name, () => {
            this.backwardLayer.build(inputShape);
        });
        this.built = true;
    }
    computeMask(inputs, mask) {
        if (Array.isArray(mask)) {
            mask = mask[0];
        }
        let outputMask;
        if (this.returnSequences) {
            if (this.mergeMode == null) {
                outputMask = [mask, mask];
            }
            else {
                outputMask = mask;
            }
        }
        else {
            if (this.mergeMode == null) {
                outputMask = [null, null];
            }
            else {
                outputMask = null;
            }
        }
        if (this.returnState) {
            const states = this.forwardLayer.states;
            const stateMask = states.map(state => null);
            if (Array.isArray(outputMask)) {
                return outputMask.concat(stateMask).concat(stateMask);
            }
            else {
                return [outputMask].concat(stateMask).concat(stateMask);
            }
        }
        else {
            return outputMask;
        }
    }
    get trainableWeights() {
        return this.forwardLayer.trainableWeights.concat(this.backwardLayer.trainableWeights);
    }
    get nonTrainableWeights() {
        return this.forwardLayer.nonTrainableWeights.concat(this.backwardLayer.nonTrainableWeights);
    }
    // TODO(cais): Implement constraints().
    setFastWeightInitDuringBuild(value) {
        super.setFastWeightInitDuringBuild(value);
        if (this.forwardLayer != null) {
            this.forwardLayer.setFastWeightInitDuringBuild(value);
        }
        if (this.backwardLayer != null) {
            this.backwardLayer.setFastWeightInitDuringBuild(value);
        }
    }
    getConfig() {
        const config = {
            'mergeMode': this.mergeMode,
        };
        // TODO(cais): Add logic for `numConstants` once the property is added.
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        const rnnLayer = deserialize(config['layer']);
        delete config['layer'];
        // TODO(cais): Add logic for `numConstants` once the property is added.
        if (config['numConstants'] != null) {
            throw new NotImplementedError(`Deserialization of a Bidirectional layer with numConstants ` +
                `present is not supported yet.`);
        }
        // tslint:disable-next-line:no-any
        const newConfig = config;
        newConfig['layer'] = rnnLayer;
        return new cls(newConfig);
    }
}
/** @nocollapse */
Bidirectional.className = 'Bidirectional';
serialization.registerClass(Bidirectional);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoid3JhcHBlcnMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvbGF5ZXJzL3dyYXBwZXJzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUg7O0dBRUc7QUFFSCxPQUFPLEtBQUssR0FBRyxNQUFNLHVCQUF1QixDQUFDO0FBQzdDLE9BQU8sRUFBQyxhQUFhLEVBQVUsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFDbEUsT0FBTyxLQUFLLENBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUM3QyxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ3BDLE9BQU8sRUFBQyxTQUFTLEVBQUUsS0FBSyxFQUFhLGNBQWMsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBQy9FLE9BQU8sRUFBQyxtQkFBbUIsRUFBRSxVQUFVLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDMUQsT0FBTyxFQUFnQywrQkFBK0IsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBR3RHLE9BQU8sS0FBSyxhQUFhLE1BQU0sd0JBQXdCLENBQUM7QUFDeEQsT0FBTyxFQUFDLGtCQUFrQixFQUFFLG1CQUFtQixFQUFDLE1BQU0sc0JBQXNCLENBQUM7QUFHN0UsT0FBTyxFQUFDLEdBQUcsRUFBTyxlQUFlLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDdEQsT0FBTyxFQUFDLFdBQVcsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBUzVDOzs7Ozs7R0FNRztBQUNILE1BQU0sT0FBZ0IsT0FBUSxTQUFRLEtBQUs7SUFHekMsWUFBWSxJQUFzQjtRQUNoQyxxRUFBcUU7UUFDckUsMEVBQTBFO1FBQzFFLDZEQUE2RDtRQUM3RCxnRUFBZ0U7UUFDaEUsd0VBQXdFO1FBQ3hFLHdFQUF3RTtRQUN4RSxvQ0FBb0M7UUFDcEMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ1osSUFBSSxDQUFDLEtBQUssR0FBRyxJ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Q3ZFLE1BQU0sVUFBVSxHQUFHLEtBQUssQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUNyQyxNQUFNLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxVQUFVLENBQUMsQ0FBQztRQUNsQyxPQUFPLE1BQU0sQ0FBQztJQUNoQixDQUFDO0lBRUQsa0JBQWtCO0lBQ2xCLE1BQU0sQ0FBVSxVQUFVLENBQ3RCLEdBQTZDLEVBQzdDLE1BQWdDO1FBQ2xDLE1BQU0sUUFBUSxHQUNWLFdBQVcsQ0FBQyxNQUFNLENBQUMsT0FBTyxDQUE2QixDQUFRLENBQUM7UUFDcEUsT0FBTyxNQUFNLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDdkIsdUVBQXVFO1FBQ3ZFLElBQUksTUFBTSxDQUFDLGNBQWMsQ0FBQyxJQUFJLElBQUksRUFBRTtZQUNsQyxNQUFNLElBQUksbUJBQW1CLENBQ3pCLDZEQUE2RDtnQkFDN0QsK0JBQStCLENBQUMsQ0FBQztTQUN0QztRQUNELGtDQUFrQztRQUNsQyxNQUFNLFNBQVMsR0FBeUIsTUFBTSxDQUFDO1FBQy9DLFNBQVMsQ0FBQyxPQUFPLENBQUMsR0FBRyxRQUFRLENBQUM7UUFDOUIsT0FBTyxJQUFJLEdBQUcsQ0FBQyxTQUFTLENBQUMsQ0FBQztJQUM1QixDQUFDOztBQS9WRCxrQkFBa0I7QUFDWCx1QkFBUyxHQUFHLGVBQWUsQ0FBQztBQWdXckMsYUFBYSxDQUFDLGFBQWEsQ0FBQyxhQUFhLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qKlxuICogTGF5ZXJzIHRoYXQgYXVnbWVudCB0aGUgZnVuY3Rpb25hbGl0eSBvZiBhIGJhc2UgbGF5ZXIuXG4gKi9cblxuaW1wb3J0ICogYXMgdGZjIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge3NlcmlhbGl6YXRpb24sIFRlbnNvciwgdGlkeX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCAqIGFzIEsgZnJvbSAnLi4vYmFja2VuZC90ZmpzX2JhY2tlbmQnO1xuaW1wb3J0IHtuYW1lU2NvcGV9IGZyb20gJy4uL2NvbW1vbic7XG5pbXBvcnQge0lucHV0U3BlYywgTGF5ZXIsIExheWVyQXJncywgU3ltYm9saWNUZW5zb3J9IGZyb20gJy4uL2VuZ2luZS90b3BvbG9neSc7XG5pbXBvcnQge05vdEltcGxlbWVudGVkRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4uL2Vycm9ycyc7XG5pbXBvcnQge0JpZGlyZWN0aW9uYWxNZXJnZU1vZGUsIFNoYXBlLCBWQUxJRF9CSURJUkVDVElPTkFMX01FUkdFX01PREVTfSBmcm9tICcuLi9rZXJhc19mb3JtYXQvY29tbW9uJztcbmltcG9ydCB7S3dhcmdzfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge1JlZ3VsYXJpemVyRm4sIFJublN0ZXBGdW5jdGlvbn0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgZ2VuZXJpY191dGlscyBmcm9tICcuLi91dGlscy9nZW5lcmljX3V0aWxzJztcbmltcG9ydCB7Z2V0RXhhY3RseU9uZVNoYXBlLCBnZXRFeGFjdGx5T25lVGVuc29yfSBmcm9tICcuLi91dGlscy90eXBlc191dGlscyc7XG5pbXBvcnQge0xheWVyVmFyaWFibGV9IGZyb20gJy4uL3ZhcmlhYmxlcyc7XG5cbmltcG9ydCB7cm5uLCBSTk4sIHN0YW5kYXJkaXplQXJnc30gZnJvbSAnLi9yZWN1cnJlbnQnO1xuaW1wb3J0IHtkZXNlcmlhbGl6ZX0gZnJvbSAnLi9zZXJpYWxpemF0aW9uJztcblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIFdyYXBwZXJMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogVGhlIGxheWVyIHRvIGJlIHdyYXBwZWQuXG4gICAqL1xuICBsYXllcjogTGF5ZXI7XG59XG5cbi8qKlxuICogQWJzdHJhY3Qgd3JhcHBlciBiYXNlIGNsYXNzLlxuICpcbiAqIFdyYXBwZXJzIHRha2UgYW5vdGhlciBsYXllciBhbmQgYXVnbWVudCBpdCBpbiB2YXJpb3VzIHdheXMuXG4gKiBEbyBub3QgdXNlIHRoaXMgY2xhc3MgYXMgYSBsYXllciwgaXQgaXMgb25seSBhbiBhYnN0cmFjdCBiYXNlIGNsYXNzLlxuICogVHdvIHVzYWJsZSB3cmFwcGVycyBhcmUgdGhlIGBUaW1lRGlzdHJpYnV0ZWRgIGFuZCBgQmlkaXJlY3Rpb25hbGAgd3JhcHBlcnMuXG4gKi9cbmV4cG9ydCBhYnN0cmFjdCBjbGFzcyBXcmFwcGVyIGV4dGVuZHMgTGF5ZXIge1xuICByZWFkb25seSBsYXllcjogTGF5ZXI7XG5cbiAgY29uc3RydWN0b3IoYXJnczogV3JhcHBlckxheWVyQXJncykge1xuICAgIC8vIFBvcnRpbmcgTm90ZTogSW4gUHlLZXJhcywgYHNlbGYubGF5ZXJgIGlzIHNldCBwcmlvciB0byB0aGUgY2FsbGluZ1xuICAgIC8vICAgYHN1cGVyKClgLiBCdXQgd2UgY2FuJ3QgZG8gdGhhdCBoZXJlIGR1ZSB0byBUeXBlU2NyaXB0J3MgcmVzdHJpY3Rpb24uXG4gICAgLy8gICBTZWU6IGh0dHBzOi8vZ2l0aHViLmNvbS9NaWNyb3NvZnQvVHlwZVNjcmlwdC9pc3N1ZXMvODI3N1xuICAgIC8vICAgQXMgYSByZXN1bHQsIHdlIGhhdmUgdG8gYWRkIGNoZWNrcyBpbiBgZ2V0IHRyYWluYWJsZSgpYCBhbmRcbiAgICAvLyAgIGBzZXQgdHJhaW5hYmxlKClgIGJlbG93IGluIG9yZGVyIHRvIHByZXZlbnQgdXNpbmcgYHRoaXMubGF5ZXJgIHdoZW5cbiAgICAvLyAgIGl0cyB2YWx1ZSBpcyBgdW5kZWZpbmVkYC4gVGhlIHN1cGVyIGNvbnN0cnVjdG9yIGRvZXMgdXNlIHRoZSBnZXR0ZXJcbiAgICAvLyAgIGFuZCB0aGUgc2V0dGVyIG9mIGB0aGlzLmxheWVyYC5cbiAgICBzdXBlcihhcmdzKTtcbiAgICB0aGlzLmxheWVyID0gYXJncy5sYXllcjtcbiAgfVxuXG4gIG92ZXJyaWRlIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICB0aGlzLmJ1aWx0ID0gdHJ1ZTtcbiAgfVxuXG4gIC8vIFRPRE8oY2Fpcyk6IEltcGxlbWVudCBhY3Rpdml0eVJlZ3VsYXJpemVyIGdldHRlci5cblxuICBvdmVycmlkZSBnZXQgdHJhaW5hYmxlKCk6IGJvb2xlYW4ge1xuICAgIC8vIFBvcnRpbmcgTm90ZTogdGhlIGNoZWNrIG9mIGB0aGlzLmxheWVyYCBoZXJlIGlzIG5lY2Vzc2FyeSBkdWUgdG8gdGhlXG4gICAgLy8gICB3YXkgdGhlIGBjb25zdHJ1Y3RvcmAgb2YgdGhpcyBjbGFzcyBpcyB3cml0dGVuIChzZWUgUG9ydGluZyBOb3RlXG4gICAgLy8gICBhYm92ZSkuXG4gICAgaWYgKHRoaXMubGF5ZXIgIT0gbnVsbCkge1xuICAgICAgcmV0dXJuIHRoaXMubGF5ZXIudHJhaW5hYmxlO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgc2V0IHRyYWluYWJsZSh2YWx1ZTogYm9vbGVhbikge1xuICAgIC8vIFBvcnRpbmcgTm90ZTogdGhlIGNoZWNrIG9mIGB0aGlzLmxheWVyYCBoZXJlIGlzIG5lY2Vzc2FyeSBkdWUgdG8gdGhlXG4gICAgLy8gICB3YXkgdGhlIGBjb25zdHJ1Y3RvcmAgb2YgdGhpcyBjbGFzcyBpcyB3cml0dGVuIChzZWUgUG9ydGluZyBOb3RlXG4gICAgLy8gICBhYm92ZSkuXG4gICAgaWYgKHRoaXMubGF5ZXIgIT0gbnVsbCkge1xuICAgICAgdGhpcy5sYXllci50cmFpbmFibGUgPSB2YWx1ZTtcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBnZXQgdHJhaW5hYmxlV2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIHJldHVybiB0aGlzLmxheWVyLnRyYWluYWJsZVdlaWdodHM7XG4gIH1cbiAgLy8gVE9ETyhjYWlzKTogSW1wbGVtZW50IHNldHRlciBmb3IgdHJhaW5hYmxlV2VpZ2h0cy5cblxuICBvdmVycmlkZSBnZXQgbm9uVHJhaW5hYmxlV2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIHJldHVybiB0aGlzLmxheWVyLm5vblRyYWluYWJsZVdlaWdodHM7XG4gIH1cbiAgLy8gVE9ETyhjYWlzKTogSW1wbGVtZW50IHNldHRlciBmb3Igbm9uVHJhaW5hYmxlV2VpZ2h0cy5cblxuICBvdmVycmlkZSBnZXQgdXBkYXRlcygpOiBUZW5zb3JbXSB7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIHJldHVybiAodGhpcy5sYXllciBhcyBhbnkpLl91cGRhdGVzO1xuICB9XG5cbiAgLy8gVE9ETyhjYWlzKTogSW1wbGVtZW50IGdldFVwZGF0ZXNGb3IoKS5cblxuICBvdmVycmlkZSBnZXQgbG9zc2VzKCk6IFJlZ3VsYXJpemVyRm5bXSB7XG4gICAgcmV0dXJuIHRoaXMubGF5ZXIubG9zc2VzO1xuICB9XG5cbiAgLy8gVE9ETyhjYWlzKTogSW1wbGVtZW50IGdldExvc3Nlc0ZvcigpLlxuXG4gIG92ZXJyaWRlIGdldFdlaWdodHMoKTogVGVuc29yW10ge1xuICAgIHJldHVybiB0aGlzLmxheWVyLmdldFdlaWdodHMoKTtcbiAgfVxuXG4gIG92ZXJyaWRlIHNldFdlaWdodHMod2VpZ2h0czogVGVuc29yW10pOiB2b2lkIHtcbiAgICB0aGlzLmxheWVyLnNldFdlaWdodHMod2VpZ2h0cyk7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtcbiAgICAgICdsYXllcic6IHtcbiAgICAgICAgJ2NsYXNzTmFtZSc6IHRoaXMubGF5ZXIuZ2V0Q2xhc3NOYW1lKCksXG4gICAgICAgICdjb25maWcnOiB0aGlzLmxheWVyLmdldENvbmZpZygpLFxuICAgICAgfVxuICAgIH07XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG5cbiAgb3ZlcnJpZGUgc2V0RmFzdFdlaWdodEluaXREdXJpbmdCdWlsZCh2YWx1ZTogYm9vbGVhbikge1xuICAgIHN1cGVyLnNldEZhc3RXZWlnaHRJbml0RHVyaW5nQnVpbGQodmFsdWUpO1xuICAgIGlmICh0aGlzLmxheWVyICE9IG51bGwpIHtcbiAgICAgIHRoaXMubGF5ZXIuc2V0RmFzdFdlaWdodEluaXREdXJpbmdCdWlsZCh2YWx1ZSk7XG4gICAgfVxuICB9XG5cbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBmcm9tQ29uZmlnPFQgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZT4oXG4gICAgICBjbHM6IHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlQ29uc3RydWN0b3I8VD4sXG4gICAgICBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCxcbiAgICAgIGN1c3RvbU9iamVjdHMgPSB7fSBhcyBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QpOiBUIHtcbiAgICBjb25zdCBsYXllckNvbmZpZyA9IGNvbmZpZ1snbGF5ZXInXSBhcyBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Q7XG4gICAgY29uc3QgbGF5ZXIgPSBkZXNlcmlhbGl6ZShsYXllckNvbmZpZywgY3VzdG9tT2JqZWN0cykgYXMgTGF5ZXI7XG4gICAgZGVsZXRlIGNvbmZpZ1snbGF5ZXInXTtcbiAgICBjb25zdCBuZXdDb25maWcgPSB7bGF5ZXJ9O1xuICAgIE9iamVjdC5hc3NpZ24obmV3Q29uZmlnLCBjb25maWcpO1xuICAgIHJldHVybiBuZXcgY2xzKG5ld0NvbmZpZyk7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIFRpbWVEaXN0cmlidXRlZCBleHRlbmRzIFdyYXBwZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdUaW1lRGlzdHJpYnV0ZWQnO1xuICBjb25zdHJ1Y3RvcihhcmdzOiBXcmFwcGVyTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gICAgdGhpcy5zdXBwb3J0c01hc2tpbmcgPSB0cnVlO1xuICB9XG5cbiAgb3ZlcnJpZGUgYnVpbGQoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IHZvaWQge1xuICAgIGlucHV0U2hhcGUgPSBnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSk7XG4gICAgaWYgKGlucHV0U2hhcGUubGVuZ3RoIDwgMykge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYFRpbWVEaXN0cmlidXRlZCBsYXllciBleHBlY3RzIGFuIGlucHV0IHNoYXBlID49IDNELCBidXQgcmVjZWl2ZWQgYCArXG4gICAgICAgICAgYGlucHV0IHNoYXBlICR7SlNPTi5zdHJpbmdpZnkoaW5wdXRTaGFwZSl9YCk7XG4gICAgfVxuICAgIHRoaXMuaW5wdXRTcGVjID0gW3tzaGFwZTogaW5wdXRTaGFwZX1dO1xuICAgIGNvbnN0IGNoaWxkSW5wdXRTaGFwZSA9IFtpbnB1dFNoYXBlWzBdXS5jb25jYXQoaW5wdXRTaGFwZS5zbGljZSgyKSk7XG4gICAgaWYgKCF0aGlzLmxheWVyLmJ1aWx0KSB7XG4gICAgICB0aGlzLmxheWVyLmJ1aWxkKGNoaWxkSW5wdXRTaGFwZSk7XG4gICAgICB0aGlzLmxheWVyLmJ1aWx0ID0gdHJ1ZTtcbiAgICB9XG4gICAgc3VwZXIuYnVpbGQoaW5wdXRTaGFwZSk7XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIGlucHV0U2hhcGUgPSBnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSk7XG4gICAgY29uc3QgY2hpbGRJbnB1dFNoYXBlID0gW2lucHV0U2hhcGVbMF1dLmNvbmNhdChpbnB1dFNoYXBlLnNsaWNlKDIpKTtcbiAgICBjb25zdCBjaGlsZE91dHB1dFNoYXBlID1cbiAgICAgICAgdGhpcy5sYXllci5jb21wdXRlT3V0cHV0U2hhcGUoY2hpbGRJbnB1dFNoYXBlKSBhcyBTaGFwZTtcbiAgICBjb25zdCB0aW1lc3RlcHMgPSBpbnB1dFNoYXBlWzFdO1xuICAgIHJldHVybiBbY2hpbGRPdXRwdXRTaGFwZVswXSwgdGltZXN0ZXBzXS5jb25jYXQoY2hpbGRPdXRwdXRTaGFwZS5zbGljZSgxKSk7XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgLy8gVE9ETyhjYWlzKTogQWRkICd0cmFpbmluZycgYW5kICd1c2VMZWFybmluZ1BoYXNlJyB0byBrd2FyZ3MuXG4gICAgICBpbnB1dHMgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XG4gICAgICAvLyBQb3J0aW5nIE5vdGU6IEluIHRmanMtbGF5ZXJzLCBgaW5wdXRzYCBhcmUgYWx3YXlzIGNvbmNyZXRlIHRlbnNvclxuICAgICAgLy8gdmFsdWVzLiBIZW5jZSB0aGUgaW5wdXRzIGNhbid0IGhhdmUgYW4gdW5kZXRlcm1pbmVkIGZpcnN0IChiYXRjaClcbiAgICAgIC8vIGRpbWVuc2lvbiwgd2hpY2ggaXMgd2h5IHdlIGFsd2F5cyB1c2UgdGhlIEsucm5uIGFwcHJvYWNoIGhlcmUuXG4gICAgICBjb25zdCBzdGVwOiBSbm5TdGVwRnVuY3Rpb24gPSAoaW5wdXRzOiBUZW5zb3IsIHN0YXRlczogVGVuc29yW10pID0+IHtcbiAgICAgICAgLy8gVE9ETyhjYWlzKTogQWRkIHVzZUxlYXJuaW5nUGhhc2UuXG4gICAgICAgIC8vIE5PVEUoY2Fpcyk6IGBsYXllci5jYWxsYCBtYXkgcmV0dXJuIGEgbGVuZ3RoLTEgYXJyYXkgb2YgVGVuc29yIGluXG4gICAgICAgIC8vICAgc29tZSBjYXNlcyAoZS5nLiwgYGxheWVyYCBpcyBhIGBTZXF1ZW50aWFsYCBpbnN0YW5jZSksIHdoaWNoIGlzXG4gICAgICAgIC8vICAgd2h5IGBnZXRFeGFjdGx5T25lVGVuc29yYCBpcyB1c2VkIGJlbG93LlxuICAgICAgICBjb25zdCBvdXRwdXQgPSBnZXRFeGFjdGx5T25lVGVuc29yKHRoaXMubGF5ZXIuY2FsbChpbnB1dHMsIGt3YXJncykpO1xuICAgICAgICByZXR1cm4gW291dHB1dCwgW11dO1xuICAgICAgfTtcbiAgICAgIGNvbnN0IHJubk91dHB1dHMgPVxuICAgICAgICAgIHJubihzdGVwLCBpbnB1dHMsIFtdLCBmYWxzZSAvKiBnb0JhY2t3YXJkcyAqLywgbnVsbCAvKiBtYXNrICovLFxuICAgICAgICAgICAgICBudWxsIC8qIGNvbnN0YW50cyAqLywgZmFsc2UgLyogdW5yb2xsICovLFxuICAgICAgICAgICAgICB0cnVlIC8qIG5lZWRQZXJTdGVwT3V0cHV0cyAqLyk7XG4gICAgICBjb25zdCB5ID0gcm5uT3V0cHV0c1sxXTtcbiAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBhY3Rpdml0eSByZWd1bGFyaXphdGlvbi5cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCB1c2VMZWFybmluZ1BoYXNlLlxuICAgICAgcmV0dXJuIHk7XG4gICAgfSk7XG4gIH1cblxuICAvLyBUT0RPKGNhaXMpOiBJbXBsZW1lbnQgZGV0YWlsZWQgY29tcHV0ZU1hc2soKSBsb2dpYy5cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhUaW1lRGlzdHJpYnV0ZWQpO1xuXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tCaWRpcmVjdGlvbmFsTWVyZ2VNb2RlKHZhbHVlPzogc3RyaW5nKTogdm9pZCB7XG4gIGdlbmVyaWNfdXRpbHMuY2hlY2tTdHJpbmdUeXBlVW5pb25WYWx1ZShcbiAgICAgIFZBTElEX0JJRElSRUNUSU9OQUxfTUVSR0VfTU9ERVMsICdCaWRpcmVjdGlvbmFsTWVyZ2VNb2RlJywgdmFsdWUpO1xufVxuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgQmlkaXJlY3Rpb25hbExheWVyQXJncyBleHRlbmRzIFdyYXBwZXJMYXllckFyZ3Mge1xuICAvKipcbiAgICogVGhlIGluc3RhbmNlIG9mIGFuIGBSTk5gIGxheWVyIHRvIGJlIHdyYXBwZWQuXG4gICAqL1xuICBsYXllcjogUk5OO1xuXG4gIC8qKlxuICAgKiBNb2RlIGJ5IHdoaWNoIG91dHB1dHMgb2YgdGhlIGZvcndhcmQgYW5kIGJhY2t3YXJkIFJOTnMgYXJlXG4gICAqIGNvbWJpbmVkLiBJZiBgbnVsbGAgb3IgYHVuZGVmaW5lZGAsIHRoZSBvdXRwdXQgd2lsbCBub3QgYmVcbiAgICogY29tYmluZWQsIHRoZXkgd2lsbCBiZSByZXR1cm5lZCBhcyBhbiBgQXJyYXlgLlxuICAgKlxuICAgKiBJZiBgdW5kZWZpbmVkYCAoaS5lLiwgbm90IHByb3ZpZGVkKSwgZGVmYXVsdHMgdG8gYCdjb25jYXQnYC5cbiAgICovXG4gIG1lcmdlTW9kZT86IEJpZGlyZWN0aW9uYWxNZXJnZU1vZGU7XG59XG5cbmNvbnN0IERFRkFVTFRfQklESVJFQ1RJT05BTF9NRVJHRV9NT0RFOiBCaWRpcmVjdGlvbmFsTWVyZ2VNb2RlID0gJ2NvbmNhdCc7XG5cbmV4cG9ydCBjbGFzcyBCaWRpcmVjdGlvbmFsIGV4dGVuZHMgV3JhcHBlciB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ0JpZGlyZWN0aW9uYWwnO1xuICBtZXJnZU1vZGU6IEJpZGlyZWN0aW9uYWxNZXJnZU1vZGU7XG4gIHByaXZhdGUgZm9yd2FyZExheWVyOiBSTk47XG4gIHByaXZhdGUgYmFja3dhcmRMYXllcjogUk5OO1xuICBwcml2YXRlIHJldHVyblNlcXVlbmNlczogYm9vbGVhbjtcbiAgcHJpdmF0ZSByZXR1cm5TdGF0ZTogYm9vbGVhbjtcbiAgcHJpdmF0ZSBudW1Db25zdGFudHM/OiBudW1iZXI7XG4gIHByaXZhdGUgX3RyYWluYWJsZTogYm9vbGVhbjtcblxuICBjb25zdHJ1Y3RvcihhcmdzOiBCaWRpcmVjdGlvbmFsTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG5cbiAgICAvLyBOb3RlOiBXaGVuIGNyZWF0aW5nIGB0aGlzLmZvcndhcmRMYXllcmAsIHRoZSBvcmlnaW5hbCBMYXllciBvYmplY3RcbiAgICAvLyAgIChgY29uZmlnLmxheWVyYCkgb3VnaHQgdG8gYmUgY2xvbmVkLiBUaGlzIGlzIHdoeSB3ZSBjYWxsXG4gICAgLy8gICBgZ2V0Q29uZmlnKClgIGZvbGxvd2VkIGJ5IGBkZXNlcmlhbGl6ZSgpYC4gV2l0aG91dCB0aGlzIGNsb25pbmcsXG4gICAgLy8gICB0aGUgbGF5ZXIgbmFtZXMgc2F2ZWQgZHVyaW5nIHNlcmlhbGl6YXRpb24gd2lsbCBpbmNvcnJlY3RseSBjb250YWluXG4gICAgLy8gICB0aGUgJ2ZvcndhcmRfJyBwcmVmaXguIEluIFB5dGhvbiBLZXJhcywgdGhpcyBpcyBkb25lIHVzaW5nXG4gICAgLy8gICBgY29weS5jb3B5YCAoc2hhbGxvdyBjb3B5KSwgd2hpY2ggZG9lcyBub3QgaGF2ZSBhIHNpbXBsZSBlcXVpdmFsZW50XG4gICAgLy8gICBpbiBKYXZhU2NyaXB0LiBKYXZhU2NyaXB0J3MgYE9iamVjdC5hc3NpZ24oKWAgZG9lcyBub3QgY29weVxuICAgIC8vICAgbWV0aG9kcy5cbiAgICBjb25zdCBsYXllckNvbmZpZyA9IGFyZ3MubGF5ZXIuZ2V0Q29uZmlnKCk7XG4gICAgY29uc3QgZm9yd0RpY3Q6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHt9O1xuICAgIGZvcndEaWN0WydjbGFzc05hbWUnXSA9IGFyZ3MubGF5ZXIuZ2V0Q2xhc3NOYW1lKCk7XG4gICAgZm9yd0RpY3RbJ2NvbmZpZyddID0gbGF5ZXJDb25maWc7XG4gICAgdGhpcy5mb3J3YXJkTGF5ZXIgPSBkZXNlcmlhbGl6ZShmb3J3RGljdCkgYXMgUk5OO1xuICAgIGxheWVyQ29uZmlnWydnb0JhY2t3YXJkcyddID1cbiAgICAgICAgbGF5ZXJDb25maWdbJ2dvQmFja3dhcmRzJ10gPT09IHRydWUgPyBmYWxzZSA6IHRydWU7XG4gICAgY29uc3QgYmFja0RpY3Q6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHt9O1xuICAgIGJhY2tEaWN0WydjbGFzc05hbWUnXSA9IGFyZ3MubGF5ZXIuZ2V0Q2xhc3NOYW1lKCk7XG4gICAgYmFja0RpY3RbJ2NvbmZpZyddID0gbGF5ZXJDb25maWc7XG4gICAgdGhpcy5iYWNrd2FyZExheWVyID0gZGVzZXJpYWxpemUoYmFja0RpY3QpIGFzIFJOTjtcbiAgICB0aGlzLmZvcndhcmRMYXllci5uYW1lID0gJ2ZvcndhcmRfJyArIHRoaXMuZm9yd2FyZExheWVyLm5hbWU7XG4gICAgdGhpcy5iYWNrd2FyZExheWVyLm5hbWUgPSAnYmFja3dhcmRfJyArIHRoaXMuYmFja3dhcmRMYXllci5uYW1lO1xuXG4gICAgdGhpcy5tZXJnZU1vZGUgPSBhcmdzLm1lcmdlTW9kZSA9PT0gdW5kZWZpbmVkID9cbiAgICAgICAgREVGQVVMVF9CSURJUkVDVElPTkFMX01FUkdFX01PREUgOlxuICAgICAgICBhcmdzLm1lcmdlTW9kZTtcbiAgICBjaGVja0JpZGlyZWN0aW9uYWxNZXJnZU1vZGUodGhpcy5tZXJnZU1vZGUpO1xuICAgIGlmIChhcmdzLndlaWdodHMpIHtcbiAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgICd3ZWlnaHRzIHN1cHBvcnQgaXMgbm90IGltcGxlbWVudGVkIGZvciBCaWRpcmVjdGlvbmFsIGxheWVyIHlldC4nKTtcbiAgICB9XG4gICAgdGhpcy5fc3RhdGVmdWwgPSBhcmdzLmxheWVyLnN0YXRlZnVsO1xuICAgIHRoaXMucmV0dXJuU2VxdWVuY2VzID0gYXJncy5sYXllci5yZXR1cm5TZXF1ZW5jZXM7XG4gICAgdGhpcy5yZXR1cm5TdGF0ZSA9IGFyZ3MubGF5ZXIucmV0dXJuU3RhdGU7XG4gICAgdGhpcy5zdXBwb3J0c01hc2tpbmcgPSB0cnVlO1xuICAgIHRoaXMuX3RyYWluYWJsZSA9IHRydWU7XG4gICAgdGhpcy5pbnB1dFNwZWMgPSBhcmdzLmxheWVyLmlucHV0U3BlYztcbiAgICB0aGlzLm51bUNvbnN0YW50cyA9IG51bGw7XG4gIH1cblxuICBvdmVycmlkZSBnZXQgdHJhaW5hYmxlKCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiB0aGlzLl90cmFpbmFibGU7XG4gIH1cblxuICBvdmVycmlkZSBzZXQgdHJhaW5hYmxlKHZhbHVlOiBib29sZWFuKSB7XG4gICAgLy8gUG9ydGluZyBOb3RlOiB0aGUgY2hlY2sgb2YgYHRoaXMubGF5ZXJgIGhlcmUgaXMgbmVjZXNzYXJ5IGR1ZSB0byB0aGVcbiAgICAvLyAgIHdheSB0aGUgYGNvbnN0cnVjdG9yYCBvZiB0aGlzIGNsYXNzIGlzIHdyaXR0ZW4gKHNlZSBQb3J0aW5nIE5vdGVcbiAgICAvLyAgIGFib3ZlKS5cbiAgICB0aGlzLl90cmFpbmFibGUgPSB2YWx1ZTtcbiAgICBpZiAodGhpcy5mb3J3YXJkTGF5ZXIgIT0gbnVsbCkge1xuICAgICAgdGhpcy5mb3J3YXJkTGF5ZXIudHJhaW5hYmxlID0gdmFsdWU7XG4gICAgfVxuICAgIGlmICh0aGlzLmJhY2t3YXJkTGF5ZXIgIT0gbnVsbCkge1xuICAgICAgdGhpcy5iYWNrd2FyZExheWVyLnRyYWluYWJsZSA9IHZhbHVlO1xuICAgIH1cbiAgfVxuXG4gIG92ZXJyaWRlIGdldFdlaWdodHMoKTogVGVuc29yW10ge1xuICAgIHJldHVybiB0aGlzLmZvcndhcmRMYXllci5nZXRXZWlnaHRzKCkuY29uY2F0KFxuICAgICAgICB0aGlzLmJhY2t3YXJkTGF5ZXIuZ2V0V2VpZ2h0cygpKTtcbiAgfVxuXG4gIG92ZXJyaWRlIHNldFdlaWdodHMod2VpZ2h0czogVGVuc29yW10pOiB2b2lkIHtcbiAgICBjb25zdCBudW1XZWlnaHRzID0gd2VpZ2h0cy5sZW5ndGg7XG4gICAgY29uc3QgbnVtZWlnaHRzT3ZlcjIgPSBNYXRoLmZsb29yKG51bVdlaWdodHMgLyAyKTtcbiAgICB0aGlzLmZvcndhcmRMYXllci5zZXRXZWlnaHRzKHdlaWdodHMuc2xpY2UoMCwgbnVtZWlnaHRzT3ZlcjIpKTtcbiAgICB0aGlzLmJhY2t3YXJkTGF5ZXIuc2V0V2VpZ2h0cyh3ZWlnaHRzLnNsaWNlKG51bWVpZ2h0c092ZXIyKSk7XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIGxldCBsYXllclNoYXBlczogU2hhcGV8U2hhcGVbXSA9XG4gICAgICAgIHRoaXMuZm9yd2FyZExheWVyLmNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBpZiAoIShBcnJheS5pc0FycmF5KGxheWVyU2hhcGVzKSAmJiBBcnJheS5pc0FycmF5KGxheWVyU2hhcGVzWzBdKSkpIHtcbiAgICAgIGxheWVyU2hhcGVzID0gW2xheWVyU2hhcGVzIGFzIFNoYXBlXTtcbiAgICB9XG4gICAgbGF5ZXJTaGFwZXMgPSBsYXllclNoYXBlcyBhcyBTaGFwZVtdO1xuXG4gICAgbGV0IG91dHB1dFNoYXBlOiBTaGFwZTtcbiAgICBsZXQgb3V0cHV0U2hhcGVzOiBTaGFwZVtdO1xuICAgIGxldCBzdGF0ZVNoYXBlOiBTaGFwZVtdO1xuICAgIGlmICh0aGlzLnJldHVyblN0YXRlKSB7XG4gICAgICBzdGF0ZVNoYXBlID0gbGF5ZXJTaGFwZXMuc2xpY2UoMSk7XG4gICAgICBvdXRwdXRTaGFwZSA9IGxheWVyU2hhcGVzWzBdO1xuICAgIH0gZWxzZSB7XG4gICAgICBvdXRwdXRTaGFwZSA9IGxheWVyU2hhcGVzWzBdO1xuICAgIH1cbiAgICBvdXRwdXRTaGFwZSA9IG91dHB1dFNoYXBlO1xuICAgIGlmICh0aGlzLm1lcmdlTW9kZSA9PT0gJ2NvbmNhdCcpIHtcbiAgICAgIG91dHB1dFNoYXBlW291dHB1dFNoYXBlLmxlbmd0aCAtIDFdICo9IDI7XG4gICAgICBvdXRwdXRTaGFwZXMgPSBbb3V0cHV0U2hhcGVdO1xuICAgIH0gZWxzZSBpZiAodGhpcy5tZXJnZU1vZGUgPT0gbnVsbCkge1xuICAgICAgb3V0cHV0U2hhcGVzID0gW291dHB1dFNoYXBlLCBvdXRwdXRTaGFwZS5zbGljZSgpXTtcbiAgICB9IGVsc2Uge1xuICAgICAgb3V0cHV0U2hhcGVzID0gW291dHB1dFNoYXBlXTtcbiAgICB9XG5cbiAgICBpZiAodGhpcy5yZXR1cm5TdGF0ZSkge1xuICAgICAgaWYgKHRoaXMubWVyZ2VNb2RlID09IG51bGwpIHtcbiAgICAgICAgcmV0dXJuIG91dHB1dFNoYXBlcy5jb25jYXQoc3RhdGVTaGFwZSkuY29uY2F0KHN0YXRlU2hhcGUuc2xpY2UoKSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gW291dHB1dFNoYXBlXS5jb25jYXQoc3RhdGVTaGFwZSkuY29uY2F0KHN0YXRlU2hhcGUuc2xpY2UoKSk7XG4gICAgfVxuICAgIHJldHVybiBnZW5lcmljX3V0aWxzLnNpbmdsZXRvbk9yQXJyYXkob3V0cHV0U2hhcGVzKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFwcGx5KFxuICAgICAgaW5wdXRzOiBUZW5zb3J8VGVuc29yW118U3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXSxcbiAgICAgIGt3YXJncz86IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcnxTeW1ib2xpY1RlbnNvcltdIHtcbiAgICBsZXQgaW5pdGlhbFN0YXRlOiBUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcltdID1cbiAgICAgICAga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydpbml0aWFsU3RhdGUnXTtcbiAgICBsZXQgY29uc3RhbnRzOiBUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcltdID1cbiAgICAgICAga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydjb25zdGFudHMnXTtcbiAgICBpZiAoa3dhcmdzID09IG51bGwpIHtcbiAgICAgIGt3YXJncyA9IHt9O1xuICAgIH1cbiAgICBjb25zdCBzdGFuZGFyZGl6ZWQgPVxuICAgICAgICBzdGFuZGFyZGl6ZUFyZ3MoaW5wdXRzLCBpbml0aWFsU3RhdGUsIGNvbnN0YW50cywgdGhpcy5udW1Db25zdGFudHMpO1xuICAgIGlucHV0cyA9IHN0YW5kYXJkaXplZC5pbnB1dHM7XG4gICAgaW5pdGlhbFN0YXRlID0gc3RhbmRhcmRpemVkLmluaXRpYWxTdGF0ZTtcbiAgICBjb25zdGFudHMgPSBzdGFuZGFyZGl6ZWQuY29uc3RhbnRzO1xuXG4gICAgaWYgKEFycmF5LmlzQXJyYXkoaW5wdXRzKSkge1xuICAgICAgaW5pdGlhbFN0YXRlID0gKGlucHV0cyBhcyBUZW5zb3JbXSB8IFN5bWJvbGljVGVuc29yW10pLnNsaWNlKDEpO1xuICAgICAgaW5wdXRzID0gKGlucHV0cyBhcyBUZW5zb3JbXSB8IFN5bWJvbGljVGVuc29yW10pWzBdO1xuICAgIH1cblxuICAgIGlmICgoaW5pdGlhbFN0YXRlID09IG51bGwgfHwgaW5pdGlhbFN0YXRlLmxlbmd0aCA9PT0gMCkgJiZcbiAgICAgICAgY29uc3RhbnRzID09IG51bGwpIHtcbiAgICAgIHJldHVybiBzdXBlci5hcHBseShpbnB1dHMsIGt3YXJncyk7XG4gICAgfVxuICAgIGNvbnN0IGFkZGl0aW9uYWxJbnB1dHM6IEFycmF5PFRlbnNvcnxTeW1ib2xpY1RlbnNvcj4gPSBbXTtcbiAgICBjb25zdCBhZGRpdGlvbmFsU3BlY3M6IElucHV0U3BlY1tdID0gW107XG4gICAgaWYgKGluaXRpYWxTdGF0ZSAhPSBudWxsKSB7XG4gICAgICBjb25zdCBudW1TdGF0ZXMgPSBpbml0aWFsU3RhdGUubGVuZ3RoO1xuICAgICAgaWYgKG51bVN0YXRlcyAlIDIgPiAwKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgJ1doZW4gcGFzc2luZyBgaW5pdGlhbFN0YXRlYCB0byBhIEJpZHJlY3Rpb25hbCBSTk4sICcgK1xuICAgICAgICAgICAgJ3RoZSBzdGF0ZSBzaG91bGQgYmUgYW4gQXJyYXkgY29udGFpbmluZyB0aGUgc3RhdGVzIG9mICcgK1xuICAgICAgICAgICAgJ3RoZSB1bmRlcmx5aW5nIFJOTnMuJyk7XG4gICAgICB9XG4gICAgICBrd2FyZ3NbJ2luaXRpYWxTdGF0ZSddID0gaW5pdGlhbFN0YXRlO1xuICAgICAgYWRkaXRpb25hbElucHV0cy5wdXNoKC4uLmluaXRpYWxTdGF0ZSk7XG4gICAgICBjb25zdCBzdGF0ZVNwZWNzID0gKGluaXRpYWxTdGF0ZSBhcyBBcnJheTxUZW5zb3J8U3ltYm9saWNUZW5zb3I+KVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAubWFwKHN0YXRlID0+IG5ldyBJbnB1dFNwZWMoe3NoYXBlOiBzdGF0ZS5zaGFwZX0pKTtcbiAgICAgIHRoaXMuZm9yd2FyZExheWVyLnN0YXRlU3BlYyA9IHN0YXRlU3BlY3Muc2xpY2UoMCwgbnVtU3RhdGVzIC8gMik7XG4gICAgICB0aGlzLmJhY2t3YXJkTGF5ZXIuc3RhdGVTcGVjID0gc3RhdGVTcGVjcy5zbGljZShudW1TdGF0ZXMgLyAyKTtcbiAgICAgIGFkZGl0aW9uYWxTcGVjcy5wdXNoKC4uLnN0YXRlU3BlY3MpO1xuICAgIH1cbiAgICBpZiAoY29uc3RhbnRzICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgICdTdXBwb3J0IGZvciBjb25zdGFudHMgaW4gQmlkaXJlY3Rpb25hbCBsYXllcnMgaXMgbm90ICcgK1xuICAgICAgICAgICdpbXBsZW1lbnRlZCB5ZXQuJyk7XG4gICAgfVxuXG4gICAgY29uc3QgaXNTeW1ib2xpY1RlbnNvciA9IGFkZGl0aW9uYWxJbnB1dHNbMF0gaW5zdGFuY2VvZiBTeW1ib2xpY1RlbnNvcjtcbiAgICBmb3IgKGNvbnN0IHRlbnNvciBvZiBhZGRpdGlvbmFsSW5wdXRzKSB7XG4gICAgICBpZiAodGVuc29yIGluc3RhbmNlb2YgU3ltYm9saWNUZW5zb3IgIT09IGlzU3ltYm9saWNUZW5zb3IpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAnVGhlIGluaXRpYWwgc3RhdGUgb2YgYSBCaWRpcmVjdGlvbmFsIGxheWVyIGNhbm5vdCBiZSAnICtcbiAgICAgICAgICAgICdzcGVjaWZpZWQgYXMgYSBtaXggb2Ygc3ltYm9saWMgYW5kIG5vbi1zeW1ib2xpYyB0ZW5zb3JzJyk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKGlzU3ltYm9saWNUZW5zb3IpIHtcbiAgICAgIC8vIENvbXB1dGUgdGhlIGZ1bGwgaW5wdXQgYW5kIHNwZWNzLCBpbmNsdWRpbmcgdGhlIHN0YXRlcy5cbiAgICAgIGNvbnN0IGZ1bGxJbnB1dCA9IFtpbnB1dHNdLmNvbmNhdChhZGRpdGlvbmFsSW5wdXRzKTtcbiAgICAgIGNvbnN0IGZ1bGxJbnB1dFNwZWMgPSB0aGlzLmlucHV0U3BlYy5jb25jYXQoYWRkaXRpb25hbFNwZWNzKTtcbiAgICAgIC8vIFBlcmZvcm0gdGhlIGNhbGwgdGVtcG9yYXJpbHkgYW5kIHJlcGxhY2UgaW5wdXRTcGVjLlxuICAgICAgLy8gTm90ZTogd2l0aCBpbml0aWFsIHN0YXRlcyBzeW1ib2xpYyBjYWxscyBhbmQgbm9uLXN5bWJvbGljIGNhbGxzIHRvXG4gICAgICAvLyB0aGlzIG1ldGhvZCBkaWZmZXIgaW4gaG93IHRoZSBpbml0aWFsIHN0YXRlcyBhcmUgcGFzc2VkLiBGb3JcbiAgICAgIC8vIHN5bWJvbGljIGNhbGxzLCB0aGUgaW5pdGlhbCBzdGF0ZXMgYXJlIHBhc3NlZCBpbiB0aGUgZmlyc3QgYXJnLCBhc1xuICAgICAgLy8gYW4gQXJyYXkgb2YgU3ltYm9saWNUZW5zb3JzOyBmb3Igbm9uLXN5bWJvbGljIGNhbGxzLCB0aGV5IGFyZVxuICAgICAgLy8gcGFzc2VkIGluIHRoZSBzZWNvbmQgYXJnIGFzIGEgcGFydCBvZiB0aGUga3dhcmdzLiBIZW5jZSB0aGUgbmVlZCB0b1xuICAgICAgLy8gdGVtcG9yYXJpbHkgbW9kaWZ5IGlucHV0U3BlYyBoZXJlLlxuICAgICAgLy8gVE9ETyhjYWlzKTogTWFrZSByZWZhY3RvcmluZyBzbyB0aGF0IHRoaXMgaGFja3kgY29kZSBiZWxvdyBpcyBub1xuICAgICAgLy8gbG9uZ2VyIG5lZWRlZC5cbiAgICAgIGNvbnN0IG9yaWdpbmFsSW5wdXRTcGVjID0gdGhpcy5pbnB1dFNwZWM7XG4gICAgICB0aGlzLmlucHV0U3BlYyA9IGZ1bGxJbnB1dFNwZWM7XG4gICAgICBjb25zdCBvdXRwdXQgPVxuICAgICAgICAgIHN1cGVyLmFwcGx5KGZ1bGxJbnB1dCBhcyBUZW5zb3JbXSB8IFN5bWJvbGljVGVuc29yW10sIGt3YXJncyk7XG4gICAgICB0aGlzLmlucHV0U3BlYyA9IG9yaWdpbmFsSW5wdXRTcGVjO1xuICAgICAgcmV0dXJuIG91dHB1dDtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHN1cGVyLmFwcGx5KGlucHV0cywga3dhcmdzKTtcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgaW5pdGlhbFN0YXRlID0ga3dhcmdzWydpbml0aWFsU3RhdGUnXTtcblxuICAgICAgbGV0IHk6IFRlbnNvcnxUZW5zb3JbXTtcbiAgICAgIGxldCB5UmV2OiBUZW5zb3J8VGVuc29yW107XG4gICAgICBpZiAoaW5pdGlhbFN0YXRlID09IG51bGwpIHtcbiAgICAgICAgeSA9IHRoaXMuZm9yd2FyZExheWVyLmNhbGwoaW5wdXRzLCBrd2FyZ3MpO1xuICAgICAgICB5UmV2ID0gdGhpcy5iYWNrd2FyZExheWVyLmNhbGwoaW5wdXRzLCBrd2FyZ3MpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgY29uc3QgZm9yd2FyZFN0YXRlID0gaW5pdGlhbFN0YXRlLnNsaWNlKDAsIGluaXRpYWxTdGF0ZS5sZW5ndGggLyAyKTtcbiAgICAgICAgY29uc3QgYmFja3dhcmRTdGF0ZSA9IGluaXRpYWxTdGF0ZS5zbGljZShpbml0aWFsU3RhdGUubGVuZ3RoIC8gMik7XG4gICAgICAgIHkgPSB0aGlzLmZvcndhcmRMYXllci5jYWxsKFxuICAgICAgICAgICAgaW5wdXRzLCBPYmplY3QuYXNzaWduKGt3YXJncywge2luaXRpYWxTdGF0ZTogZm9yd2FyZFN0YXRlfSkpO1xuICAgICAgICB5UmV2ID0gdGhpcy5iYWNrd2FyZExheWVyLmNhbGwoXG4gICAgICAgICAgICBpbnB1dHMsIE9iamVjdC5hc3NpZ24oa3dhcmdzLCB7aW5pdGlhbFN0YXRlOiBiYWNrd2FyZFN0YXRlfSkpO1xuICAgICAgfVxuXG4gICAgICBsZXQgc3RhdGVzOiBUZW5zb3JbXTtcbiAgICAgIGlmICh0aGlzLnJldHVyblN0YXRlKSB7XG4gICAgICAgIGlmIChBcnJheS5pc0FycmF5KHkpKSB7XG4gICAgICAgICAgc3RhdGVzID0geS5zbGljZSgxKS5jb25jYXQoKHlSZXYgYXMgVGVuc29yW10pLnNsaWNlKDEpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgfVxuICAgICAgICB5ID0gKHkgYXMgVGVuc29yW10pWzBdO1xuICAgICAgICB5UmV2ID0gKHlSZXYgYXMgVGVuc29yW10pWzBdO1xuICAgICAgfVxuXG4gICAgICBpZiAodGhpcy5yZXR1cm5TZXF1ZW5jZXMpIHtcbiAgICAgICAgeVJldiA9IHRmYy5yZXZlcnNlKHlSZXYgYXMgVGVuc29yLCAxKTtcbiAgICAgIH1cblxuICAgICAgbGV0IG91dHB1dDogVGVuc29yfFRlbnNvcltdO1xuICAgICAgaWYgKHRoaXMubWVyZ2VNb2RlID09PSAnY29uY2F0Jykge1xuICAgICAgICBvdXRwdXQgPSBLLmNvbmNhdGVuYXRlKFt5IGFzIFRlbnNvciwgeVJldiBhcyBUZW5zb3JdKTtcbiAgICAgIH0gZWxzZSBpZiAodGhpcy5tZXJnZU1vZGUgPT09ICdzdW0nKSB7XG4gICAgICAgIG91dHB1dCA9IHRmYy5hZGQoeSBhcyBUZW5zb3IsIHlSZXYgYXMgVGVuc29yKTtcbiAgICAgIH0gZWxzZSBpZiAodGhpcy5tZXJnZU1vZGUgPT09ICdhdmUnKSB7XG4gICAgICAgIG91dHB1dCA9IHRmYy5tdWwoLjUsIHRmYy5hZGQoeSBhcyBUZW5zb3IsIHlSZXYgYXMgVGVuc29yKSk7XG4gICAgICB9IGVsc2UgaWYgKHRoaXMubWVyZ2VNb2RlID09PSAnbXVsJykge1xuICAgICAgICBvdXRwdXQgPSB0ZmMubXVsKHkgYXMgVGVuc29yLCB5UmV2IGFzIFRlbnNvcik7XG4gICAgICB9IGVsc2UgaWYgKHRoaXMubWVyZ2VNb2RlID09IG51bGwpIHtcbiAgICAgICAgb3V0cHV0ID0gW3kgYXMgVGVuc29yLCB5UmV2IGFzIFRlbnNvcl07XG4gICAgICB9XG5cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IFByb3Blcmx5IHNldCBsZWFybmluZyBwaGFzZS5cbiAgICAgIGlmICh0aGlzLnJldHVyblN0YXRlKSB7XG4gICAgICAgIGlmICh0aGlzLm1lcmdlTW9kZSA9PSBudWxsKSB7XG4gICAgICAgICAgcmV0dXJuIChvdXRwdXQgYXMgVGVuc29yW10pLmNvbmNhdChzdGF0ZXMpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBbb3V0cHV0IGFzIFRlbnNvcl0uY29uY2F0KHN0YXRlcyk7XG4gICAgICB9XG4gICAgICByZXR1cm4gb3V0cHV0O1xuICAgIH0pO1xuICB9XG5cbiAgb3ZlcnJpZGUgcmVzZXRTdGF0ZXMoc3RhdGVzPzogVGVuc29yfFRlbnNvcltdKTogdm9pZCB7XG4gICAgdGhpcy5mb3J3YXJkTGF5ZXIucmVzZXRTdGF0ZXMoKTtcbiAgICB0aGlzLmJhY2t3YXJkTGF5ZXIucmVzZXRTdGF0ZXMoKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICBuYW1lU2NvcGUodGhpcy5mb3J3YXJkTGF5ZXIubmFtZSwgKCkgPT4ge1xuICAgICAgdGhpcy5mb3J3YXJkTGF5ZXIuYnVpbGQoaW5wdXRTaGFwZSk7XG4gICAgfSk7XG4gICAgbmFtZVNjb3BlKHRoaXMuYmFja3dhcmRMYXllci5uYW1lLCAoKSA9PiB7XG4gICAgICB0aGlzLmJhY2t3YXJkTGF5ZXIuYnVpbGQoaW5wdXRTaGFwZSk7XG4gICAgfSk7XG4gICAgdGhpcy5idWlsdCA9IHRydWU7XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlTWFzayhpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwgbWFzaz86IFRlbnNvcnxUZW5zb3JbXSk6IFRlbnNvclxuICAgICAgfFRlbnNvcltdIHtcbiAgICBpZiAoQXJyYXkuaXNBcnJheShtYXNrKSkge1xuICAgICAgbWFzayA9IG1hc2tbMF07XG4gICAgfVxuICAgIGxldCBvdXRwdXRNYXNrOiBUZW5zb3J8VGVuc29yW107XG4gICAgaWYgKHRoaXMucmV0dXJuU2VxdWVuY2VzKSB7XG4gICAgICBpZiAodGhpcy5tZXJnZU1vZGUgPT0gbnVsbCkge1xuICAgICAgICBvdXRwdXRNYXNrID0gW21hc2ssIG1hc2tdO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgb3V0cHV0TWFzayA9IG1hc2s7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGlmICh0aGlzLm1lcmdlTW9kZSA9PSBudWxsKSB7XG4gICAgICAgIG91dHB1dE1hc2sgPSBbbnVsbCwgbnVsbF07XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBvdXRwdXRNYXNrID0gbnVsbDtcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKHRoaXMucmV0dXJuU3RhdGUpIHtcbiAgICAgIGNvbnN0IHN0YXRlcyA9IHRoaXMuZm9yd2FyZExheWVyLnN0YXRlcztcbiAgICAgIGNvbnN0IHN0YXRlTWFzazogVGVuc29yW10gPSBzdGF0ZXMubWFwKHN0YXRlID0+IG51bGwpO1xuICAgICAgaWYgKEFycmF5LmlzQXJyYXkob3V0cHV0TWFzaykpIHtcbiAgICAgICAgcmV0dXJuIG91dHB1dE1hc2suY29uY2F0KHN0YXRlTWFzaykuY29uY2F0KHN0YXRlTWFzayk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gW291dHB1dE1hc2tdLmNvbmNhdChzdGF0ZU1hc2spLmNvbmNhdChzdGF0ZU1hc2spO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gb3V0cHV0TWFzaztcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBnZXQgdHJhaW5hYmxlV2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIHJldHVybiB0aGlzLmZvcndhcmRMYXllci50cmFpbmFibGVXZWlnaHRzLmNvbmNhdChcbiAgICAgICAgdGhpcy5iYWNrd2FyZExheWVyLnRyYWluYWJsZVdlaWdodHMpO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0IG5vblRyYWluYWJsZVdlaWdodHMoKTogTGF5ZXJWYXJpYWJsZVtdIHtcbiAgICByZXR1cm4gdGhpcy5mb3J3YXJkTGF5ZXIubm9uVHJhaW5hYmxlV2VpZ2h0cy5jb25jYXQoXG4gICAgICAgIHRoaXMuYmFja3dhcmRMYXllci5ub25UcmFpbmFibGVXZWlnaHRzKTtcbiAgfVxuXG4gIC8vIFRPRE8oY2Fpcyk6IEltcGxlbWVudCBjb25zdHJhaW50cygpLlxuXG4gIG92ZXJyaWRlIHNldEZhc3RXZWlnaHRJbml0RHVyaW5nQnVpbGQodmFsdWU6IGJvb2xlYW4pIHtcbiAgICBzdXBlci5zZXRGYXN0V2VpZ2h0SW5pdER1cmluZ0J1aWxkKHZhbHVlKTtcbiAgICBpZiAodGhpcy5mb3J3YXJkTGF5ZXIgIT0gbnVsbCkge1xuICAgICAgdGhpcy5mb3J3YXJkTGF5ZXIuc2V0RmFzdFdlaWdodEluaXREdXJpbmdCdWlsZCh2YWx1ZSk7XG4gICAgfVxuICAgIGlmICh0aGlzLmJhY2t3YXJkTGF5ZXIgIT0gbnVsbCkge1xuICAgICAgdGhpcy5iYWNrd2FyZExheWVyLnNldEZhc3RXZWlnaHRJbml0RHVyaW5nQnVpbGQodmFsdWUpO1xuICAgIH1cbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge1xuICAgICAgJ21lcmdlTW9kZSc6IHRoaXMubWVyZ2VNb2RlLFxuICAgIH07XG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIGxvZ2ljIGZvciBgbnVtQ29uc3RhbnRzYCBvbmNlIHRoZSBwcm9wZXJ0eSBpcyBhZGRlZC5cbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cblxuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIG92ZXJyaWRlIGZyb21Db25maWc8VCBleHRlbmRzIHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlPihcbiAgICAgIGNsczogc2VyaWFsaXphdGlvbi5TZXJpYWxpemFibGVDb25zdHJ1Y3RvcjxUPixcbiAgICAgIGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0KTogVCB7XG4gICAgY29uc3Qgcm5uTGF5ZXIgPVxuICAgICAgICBkZXNlcmlhbGl6ZShjb25maWdbJ2xheWVyJ10gYXMgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0KSBhcyBSTk47XG4gICAgZGVsZXRlIGNvbmZpZ1snbGF5ZXInXTtcbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgbG9naWMgZm9yIGBudW1Db25zdGFudHNgIG9uY2UgdGhlIHByb3BlcnR5IGlzIGFkZGVkLlxuICAgIGlmIChjb25maWdbJ251bUNvbnN0YW50cyddICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgIGBEZXNlcmlhbGl6YXRpb24gb2YgYSBCaWRpcmVjdGlvbmFsIGxheWVyIHdpdGggbnVtQ29uc3RhbnRzIGAgK1xuICAgICAgICAgIGBwcmVzZW50IGlzIG5vdCBzdXBwb3J0ZWQgeWV0LmApO1xuICAgIH1cbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgY29uc3QgbmV3Q29uZmlnOiB7W2tleTogc3RyaW5nXTogYW55fSA9IGNvbmZpZztcbiAgICBuZXdDb25maWdbJ2xheWVyJ10gPSBybm5MYXllcjtcbiAgICByZXR1cm4gbmV3IGNscyhuZXdDb25maWcpO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQmlkaXJlY3Rpb25hbCk7XG4iXX0=