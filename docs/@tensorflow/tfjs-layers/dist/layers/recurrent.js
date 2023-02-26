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
 * TensorFlow.js Layers: Recurrent Neural Network Layers.
 */
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy, util } from '@tensorflow/tfjs-core';
import { getActivation, serializeActivation } from '../activations';
import * as K from '../backend/tfjs_backend';
import { nameScope } from '../common';
import { getConstraint, serializeConstraint } from '../constraints';
import { InputSpec, SymbolicTensor } from '../engine/topology';
import { Layer } from '../engine/topology';
import { AttributeError, NotImplementedError, ValueError } from '../errors';
import { getInitializer, Initializer, Ones, serializeInitializer } from '../initializers';
import { getRegularizer, serializeRegularizer } from '../regularizers';
import { assertPositiveInteger } from '../utils/generic_utils';
import * as math_utils from '../utils/math_utils';
import { getExactlyOneShape, getExactlyOneTensor, isArrayOfShapes } from '../utils/types_utils';
import { batchGetValue, batchSetValue } from '../variables';
import { deserialize } from './serialization';
/**
 * Standardize `apply()` args to a single list of tensor inputs.
 *
 * When running a model loaded from file, the input tensors `initialState` and
 * `constants` are passed to `RNN.apply()` as part of `inputs` instead of the
 * dedicated kwargs fields. `inputs` consists of
 * `[inputs, initialState0, initialState1, ..., constant0, constant1]` in this
 * case.
 * This method makes sure that arguments are
 * separated and that `initialState` and `constants` are `Array`s of tensors
 * (or None).
 *
 * @param inputs Tensor or `Array` of  tensors.
 * @param initialState Tensor or `Array` of tensors or `null`/`undefined`.
 * @param constants Tensor or `Array` of tensors or `null`/`undefined`.
 * @returns An object consisting of
 *   inputs: A tensor.
 *   initialState: `Array` of tensors or `null`.
 *   constants: `Array` of tensors or `null`.
 * @throws ValueError, if `inputs` is an `Array` but either `initialState` or
 *   `constants` is provided.
 */
export function standardizeArgs(inputs, initialState, constants, numConstants) {
    if (Array.isArray(inputs)) {
        if (initialState != null || constants != null) {
            throw new ValueError('When inputs is an array, neither initialState or constants ' +
                'should be provided');
        }
        if (numConstants != null) {
            constants = inputs.slice(inputs.length - numConstants, inputs.length);
            inputs = inputs.slice(0, inputs.length - numConstants);
        }
        if (inputs.length > 1) {
            initialState = inputs.slice(1, inputs.length);
        }
        inputs = inputs[0];
    }
    function toListOrNull(x) {
        if (x == null || Array.isArray(x)) {
            return x;
        }
        else {
            return [x];
        }
    }
    initialState = toListOrNull(initialState);
    constants = toListOrNull(constants);
    return { inputs, initialState, constants };
}
/**
 * Iterates over the time dimension of a tensor.
 *
 * @param stepFunction RNN step function.
 *   Parameters:
 *     inputs: tensor with shape `[samples, ...]` (no time dimension),
 *       representing input for the batch of samples at a certain time step.
 *     states: an Array of tensors.
 *   Returns:
 *     outputs: tensor with shape `[samples, outputDim]` (no time dimension).
 *     newStates: list of tensors, same length and shapes as `states`. The first
 *       state in the list must be the output tensor at the previous timestep.
 * @param inputs Tensor of temporal data of shape `[samples, time, ...]` (at
 *   least 3D).
 * @param initialStates Tensor with shape `[samples, outputDim]` (no time
 *   dimension), containing the initial values of the states used in the step
 *   function.
 * @param goBackwards If `true`, do the iteration over the time dimension in
 *   reverse order and return the reversed sequence.
 * @param mask Binary tensor with shape `[sample, time, 1]`, with a zero for
 *   every element that is masked.
 * @param constants An Array of constant values passed at each step.
 * @param unroll Whether to unroll the RNN or to use a symbolic loop. *Not*
 *   applicable to this imperative deeplearn.js backend. Its value is ignored.
 * @param needPerStepOutputs Whether the per-step outputs are to be
 *   concatenated into a single tensor and returned (as the second return
 *   value). Default: `false`. This arg is included so that the relatively
 *   expensive concatenation of the stepwise outputs can be omitted unless
 *   the stepwise outputs need to be kept (e.g., for an LSTM layer of which
 *   `returnSequence` is `true`.)
 * @returns An Array: `[lastOutput, outputs, newStates]`.
 *   lastOutput: the lastest output of the RNN, of shape `[samples, ...]`.
 *   outputs: tensor with shape `[samples, time, ...]` where each entry
 *     `output[s, t]` is the output of the step function at time `t` for sample
 *     `s`. This return value is provided if and only if the
 *     `needPerStepOutputs` is set as `true`. If it is set as `false`, this
 *     return value will be `undefined`.
 *   newStates: Array of tensors, latest states returned by the step function,
 *      of shape `(samples, ...)`.
 * @throws ValueError If input dimension is less than 3.
 *
 * TODO(nielsene): This needs to be tidy-ed.
 */
export function rnn(stepFunction, inputs, initialStates, goBackwards = false, mask, constants, unroll = false, needPerStepOutputs = false) {
    return tfc.tidy(() => {
        const ndim = inputs.shape.length;
        if (ndim < 3) {
            throw new ValueError(`Input should be at least 3D, but is ${ndim}D.`);
        }
        // Transpose to time-major, i.e., from [batch, time, ...] to [time, batch,
        // ...].
        const axes = [1, 0].concat(math_utils.range(2, ndim));
        inputs = tfc.transpose(inputs, axes);
        if (constants != null) {
            throw new NotImplementedError('The rnn() functoin of the deeplearn.js backend does not support ' +
                'constants yet.');
        }
        // Porting Note: the unroll option is ignored by the imperative backend.
        if (unroll) {
            console.warn('Backend rnn(): the unroll = true option is not applicable to the ' +
                'imperative deeplearn.js backend.');
        }
        if (mask != null) {
            mask = tfc.cast(tfc.cast(mask, 'bool'), 'float32');
            if (mask.rank === ndim - 1) {
                mask = tfc.expandDims(mask, -1);
            }
            mask = tfc.transpose(mask, axes);
        }
        if (goBackwards) {
            inputs = tfc.reverse(inputs, 0);
            if (mask != null) {
                mask = tfc.reverse(mask, 0);
            }
        }
        // Porting Note: PyKeras with TensorFlow backend uses a symbolic loop
        //   (tf.while_loop). But for the imperative deeplearn.js backend, we just
        //   use the usual TypeScript control flow to iterate over the time steps in
        //   the inputs.
        // Porting Note: PyKeras patches a "_use_learning_phase" attribute to
        // outputs.
        //   This is not idiomatic in TypeScript. The info regarding whether we are
        //   in a learning (i.e., training) phase for RNN is passed in a different
        //   way.
        const perStepOutputs = [];
        let lastOutput;
        let states = initialStates;
        const timeSteps = inputs.shape[0];
        const perStepInputs = tfc.unstack(inputs);
        let perStepMasks;
        if (mask != null) {
            perStepMasks = tfc.unstack(mask);
        }
        for (let t = 0; t < timeSteps; ++t) {
            const currentInput = perStepInputs[t];
            const stepOutputs = tfc.tidy(() => stepFunction(currentInput, states));
            if (mask == null) {
                lastOutput = stepOutputs[0];
                states = stepOutputs[1];
            }
            else {
                const maskedOutputs = tfc.tidy(() => {
                    const stepMask = perStepMasks[t];
                    const negStepMask = tfc.sub(tfc.onesLike(stepMask), stepMask);
                    // TODO(cais): Would tfc.where() be better for performance?
                    const output = tfc.add(tfc.mul(stepOutputs[0], stepMask), tfc.mul(states[0], negStepMask));
                    const newStates = states.map((state, i) => {
                        return tfc.add(tfc.mul(stepOutputs[1][i], stepMask), tfc.mul(state, negStepMask));
                    });
                    return { output, newStates };
                });
                lastOutput = maskedOutputs.output;
                states = maskedOutputs.newStates;
            }
            if (needPerStepOutputs) {
                perStepOutputs.push(lastOutput);
            }
        }
        let outputs;
        if (needPerStepOutputs) {
            const axis = 1;
            outputs = tfc.stack(perStepOutputs, axis);
        }
        return [lastOutput, outputs, states];
    });
}
export class RNN extends Layer {
    constructor(args) {
        super(args);
        let cell;
        if (args.cell == null) {
            throw new ValueError('cell property is missing for the constructor of RNN.');
        }
        else if (Array.isArray(args.cell)) {
            cell = new StackedRNNCells({ cells: args.cell });
        }
        else {
            cell = args.cell;
        }
        if (cell.stateSize == null) {
            throw new ValueError('The RNN cell should have an attribute `stateSize` (tuple of ' +
                'integers, one integer per RNN state).');
        }
        this.cell = cell;
        this.returnSequences =
            args.returnSequences == null ? false : args.returnSequences;
        this.returnState = args.returnState == null ? false : args.returnState;
        this.goBackwards = args.goBackwards == null ? false : args.goBackwards;
        this._stateful = args.stateful == null ? false : args.stateful;
        this.unroll = args.unroll == null ? false : args.unroll;
        this.supportsMasking = true;
        this.inputSpec = [new InputSpec({ ndim: 3 })];
        this.stateSpec = null;
        this.states_ = null;
        // TODO(cais): Add constantsSpec and numConstants.
        this.numConstants = null;
        // TODO(cais): Look into the use of initial_state in the kwargs of the
        //   constructor.
        this.keptStates = [];
    }
    // Porting Note: This is the equivalent of `RNN.states` property getter in
    //   PyKeras.
    getStates() {
        if (this.states_ == null) {
            const numStates = Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
            return math_utils.range(0, numStates).map(x => null);
        }
        else {
            return this.states_;
        }
    }
    // Porting Note: This is the equivalent of the `RNN.states` property setter in
    //   PyKeras.
    setStates(states) {
        this.states_ = states;
    }
    computeOutputShape(inputShape) {
        if (isArrayOfShapes(inputShape)) {
            inputShape = inputShape[0];
        }
        inputShape = inputShape;
        // TODO(cais): Remove the casting once stacked RNN cells become supported.
        let stateSize = this.cell.stateSize;
        if (!Array.isArray(stateSize)) {
            stateSize = [stateSize];
        }
        const outputDim = stateSize[0];
        let outputShape;
        if (this.returnSequences) {
            outputShape = [inputShape[0], inputShape[1], outputDim];
        }
        else {
            outputShape = [inputShape[0], outputDim];
        }
        if (this.returnState) {
            const stateShape = [];
            for (const dim of stateSize) {
                stateShape.push([inputShape[0], dim]);
            }
            return [outputShape].concat(stateShape);
        }
        else {
            return outputShape;
        }
    }
    computeMask(inputs, mask) {
        return tfc.tidy(() => {
            if (Array.isArray(mask)) {
                mask = mask[0];
            }
            const outputMask = this.returnSequences ? mask : null;
            if (this.returnState) {
                const stateMask = this.states.map(s => null);
                return [outputMask].concat(stateMask);
            }
            else {
                return outputMask;
            }
        });
    }
    /**
     * Get the current state tensors of the RNN.
     *
     * If the state hasn't been set, return an array of `null`s of the correct
     * length.
     */
    get states() {
        if (this.states_ == null) {
            const numStates = Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
            const output = [];
            for (let i = 0; i < numStates; ++i) {
                output.push(null);
            }
            return output;
        }
        else {
            return this.states_;
        }
    }
    set states(s) {
        this.states_ = s;
    }
    build(inputShape) {
        // Note inputShape will be an Array of Shapes of initial states and
        // constants if these are passed in apply().
        const constantShape = null;
        if (this.numConstants != null) {
            throw new NotImplementedError('Constants support is not implemented in RNN yet.');
        }
        if (isArrayOfShapes(inputShape)) {
            inputShape = inputShape[0];
        }
        inputShape = inputShape;
        const batchSize = this.stateful ? inputShape[0] : null;
        const inputDim = inputShape.slice(2);
        this.inputSpec[0] = new InputSpec({ shape: [batchSize, null, ...inputDim] });
        // Allow cell (if RNNCell Layer) to build before we set or validate
        // stateSpec.
        const stepInputShape = [inputShape[0]].concat(inputShape.slice(2));
        if (constantShape != null) {
            throw new NotImplementedError('Constants support is not implemented in RNN yet.');
        }
        else {
            this.cell.build(stepInputShape);
        }
        // Set or validate stateSpec.
        let stateSize;
        if (Array.isArray(this.cell.stateSize)) {
            stateSize = this.cell.stateSize;
        }
        else {
            stateSize = [this.cell.stateSize];
        }
        if (this.stateSpec != null) {
            if (!util.arraysEqual(this.stateSpec.map(spec => spec.shape[spec.shape.length - 1]), stateSize)) {
                throw new ValueError(`An initialState was passed that is not compatible with ` +
                    `cell.stateSize. Received stateSpec=${this.stateSpec}; ` +
                    `However cell.stateSize is ${this.cell.stateSize}`);
            }
        }
        else {
            this.stateSpec =
                stateSize.map(dim => new InputSpec({ shape: [null, dim] }));
        }
        if (this.stateful) {
            this.resetStates();
        }
    }
    /**
     * Reset the state tensors of the RNN.
     *
     * If the `states` argument is `undefined` or `null`, will set the
     * state tensor(s) of the RNN to all-zero tensors of the appropriate
     * shape(s).
     *
     * If `states` is provided, will set the state tensors of the RNN to its
     * value.
     *
     * @param states Optional externally-provided initial states.
     * @param training Whether this call is done during training. For stateful
     *   RNNs, this affects whether the old states are kept or discarded. In
     *   particular, if `training` is `true`, the old states will be kept so
     *   that subsequent backpropgataion through time (BPTT) may work properly.
     *   Else, the old states will be discarded.
     */
    resetStates(states, training = false) {
        tidy(() => {
            if (!this.stateful) {
                throw new AttributeError('Cannot call resetStates() on an RNN Layer that is not stateful.');
            }
            const batchSize = this.inputSpec[0].shape[0];
            if (batchSize == null) {
                throw new ValueError('If an RNN is stateful, it needs to know its batch size. Specify ' +
                    'the batch size of your input tensors: \n' +
                    '- If using a Sequential model, specify the batch size by ' +
                    'passing a `batchInputShape` option to your first layer.\n' +
                    '- If using the functional API, specify the batch size by ' +
                    'passing a `batchShape` option to your Input layer.');
            }
            // Initialize state if null.
            if (this.states_ == null) {
                if (Array.isArray(this.cell.stateSize)) {
                    this.states_ =
                        this.cell.stateSize.map(dim => tfc.zeros([batchSize, dim]));
                }
                else {
                    this.states_ = [tfc.zeros([batchSize, this.cell.stateSize])];
                }
            }
            else if (states == null) {
                // Dispose old state tensors.
                tfc.dispose(this.states_);
                // For stateful RNNs, fully dispose kept old states.
                if (this.keptStates != null) {
                    tfc.dispose(this.keptStates);
                    this.keptStates = [];
                }
                if (Array.isArray(this.cell.stateSize)) {
                    this.states_ =
                        this.cell.stateSize.map(dim => tfc.zeros([batchSize, dim]));
                }
                else {
                    this.states_[0] = tfc.zeros([batchSize, this.cell.stateSize]);
                }
            }
            else {
                if (!Array.isArray(states)) {
                    states = [states];
                }
                if (states.length !== this.states_.length) {
                    throw new ValueError(`Layer ${this.name} expects ${this.states_.length} state(s), ` +
                        `but it received ${states.length} state value(s). Input ` +
                        `received: ${states}`);
                }
                if (training === true) {
                    // Store old state tensors for complete disposal later, i.e., during
                    // the next no-arg call to this method. We do not dispose the old
                    // states immediately because that BPTT (among other things) require
                    // them.
                    this.keptStates.push(this.states_.slice());
                }
                else {
                    tfc.dispose(this.states_);
                }
                for (let index = 0; index < this.states_.length; ++index) {
                    const value = states[index];
                    const dim = Array.isArray(this.cell.stateSize) ?
                        this.cell.stateSize[index] :
                        this.cell.stateSize;
                    const expectedShape = [batchSize, dim];
                    if (!util.arraysEqual(value.shape, expectedShape)) {
                        throw new ValueError(`State ${index} is incompatible with layer ${this.name}: ` +
                            `expected shape=${expectedShape}, received shape=${value.shape}`);
                    }
                    this.states_[index] = value;
                }
            }
            this.states_ = this.states_.map(state => tfc.keep(state.clone()));
        });
    }
    apply(inputs, kwargs) {
        // TODO(cais): Figure out whether initialState is in kwargs or inputs.
        let initialState = kwargs == null ? null : kwargs['initialState'];
        let constants = kwargs == null ? null : kwargs['constants'];
        if (kwargs == null) {
            kwargs = {};
        }
        const standardized = standardizeArgs(inputs, initialState, constants, this.numConstants);
        inputs = standardized.inputs;
        initialState = standardized.initialState;
        constants = standardized.constants;
        // If any of `initial_state` or `constants` are specified and are
        // `tf.SymbolicTensor`s, then add them to the inputs and temporarily modify
        // the input_spec to include them.
        let additionalInputs = [];
        let additionalSpecs = [];
        if (initialState != null) {
            kwargs['initialState'] = initialState;
            additionalInputs = additionalInputs.concat(initialState);
            this.stateSpec = [];
            for (const state of initialState) {
                this.stateSpec.push(new InputSpec({ shape: state.shape }));
            }
            // TODO(cais): Use the following instead.
            // this.stateSpec = initialState.map(state => new InputSpec({shape:
            // state.shape}));
            additionalSpecs = additionalSpecs.concat(this.stateSpec);
        }
        if (constants != null) {
            kwargs['constants'] = constants;
            additionalInputs = additionalInputs.concat(constants);
            // TODO(cais): Add this.constantsSpec.
            this.numConstants = constants.length;
        }
        const isTensor = additionalInputs[0] instanceof SymbolicTensor;
        if (isTensor) {
            // Compute full input spec, including state and constants.
            const fullInput = [inputs].concat(additionalInputs);
            const fullInputSpec = this.inputSpec.concat(additionalSpecs);
            // Perform the call with temporarily replaced inputSpec.
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
    // tslint:disable-next-line:no-any
    call(inputs, kwargs) {
        // Input shape: `[samples, time (padded with zeros), input_dim]`.
        // Note that the .build() method of subclasses **must** define
        // this.inputSpec and this.stateSpec owith complete input shapes.
        return tidy(() => {
            const mask = kwargs == null ? null : kwargs['mask'];
            const training = kwargs == null ? null : kwargs['training'];
            let initialState = kwargs == null ? null : kwargs['initialState'];
            inputs = getExactlyOneTensor(inputs);
            if (initialState == null) {
                if (this.stateful) {
                    initialState = this.states_;
                }
                else {
                    initialState = this.getInitialState(inputs);
                }
            }
            const numStates = Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
            if (initialState.length !== numStates) {
                throw new ValueError(`RNN Layer has ${numStates} state(s) but was passed ` +
                    `${initialState.length} initial state(s).`);
            }
            if (this.unroll) {
                console.warn('Ignoring unroll = true for RNN layer, due to imperative backend.');
            }
            const cellCallKwargs = { training };
            // TODO(cais): Add support for constants.
            const step = (inputs, states) => {
                // `inputs` and `states` are concatenated to form a single `Array` of
                // `tf.Tensor`s as the input to `cell.call()`.
                const outputs = this.cell.call([inputs].concat(states), cellCallKwargs);
                // Marshall the return value into output and new states.
                return [outputs[0], outputs.slice(1)];
            };
            // TODO(cais): Add support for constants.
            const rnnOutputs = rnn(step, inputs, initialState, this.goBackwards, mask, null, this.unroll, this.returnSequences);
            const lastOutput = rnnOutputs[0];
            const outputs = rnnOutputs[1];
            const states = rnnOutputs[2];
            if (this.stateful) {
                this.resetStates(states, training);
            }
            const output = this.returnSequences ? outputs : lastOutput;
            // TODO(cais): Porperty set learning phase flag.
            if (this.returnState) {
                return [output].concat(states);
            }
            else {
                return output;
            }
        });
    }
    getInitialState(inputs) {
        return tidy(() => {
            // Build an all-zero tensor of shape [samples, outputDim].
            // [Samples, timeSteps, inputDim].
            let initialState = tfc.zeros(inputs.shape);
            // [Samples].
            initialState = tfc.sum(initialState, [1, 2]);
            initialState = K.expandDims(initialState); // [Samples, 1].
            if (Array.isArray(this.cell.stateSize)) {
                return this.cell.stateSize.map(dim => dim > 1 ? K.tile(initialState, [1, dim]) : initialState);
            }
            else {
                return this.cell.stateSize > 1 ?
                    [K.tile(initialState, [1, this.cell.stateSize])] :
                    [initialState];
            }
        });
    }
    get trainableWeights() {
        if (!this.trainable) {
            return [];
        }
        // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
        return this.cell.trainableWeights;
    }
    get nonTrainableWeights() {
        // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
        if (!this.trainable) {
            return this.cell.weights;
        }
        return this.cell.nonTrainableWeights;
    }
    setFastWeightInitDuringBuild(value) {
        super.setFastWeightInitDuringBuild(value);
        if (this.cell != null) {
            this.cell.setFastWeightInitDuringBuild(value);
        }
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = {
            returnSequences: this.returnSequences,
            returnState: this.returnState,
            goBackwards: this.goBackwards,
            stateful: this.stateful,
            unroll: this.unroll,
        };
        if (this.numConstants != null) {
            config['numConstants'] = this.numConstants;
        }
        const cellConfig = this.cell.getConfig();
        if (this.getClassName() === RNN.className) {
            config['cell'] = {
                'className': this.cell.getClassName(),
                'config': cellConfig,
            };
        }
        // this order is necessary, to prevent cell name from replacing layer name
        return Object.assign(Object.assign(Object.assign({}, cellConfig), baseConfig), config);
    }
    /** @nocollapse */
    static fromConfig(cls, config, customObjects = {}) {
        const cellConfig = config['cell'];
        const cell = deserialize(cellConfig, customObjects);
        return new cls(Object.assign(config, { cell }));
    }
}
/** @nocollapse */
RNN.className = 'RNN';
serialization.registerClass(RNN);
// Porting Note: This is a common parent class for RNN cells. There is no
// equivalent of this in PyKeras. Having a common parent class forgoes the
//  need for `has_attr(cell, ...)` checks or its TypeScript equivalent.
/**
 * An RNNCell layer.
 *
 * @doc {heading: 'Layers', subheading: 'Classes'}
 */
export class RNNCell extends Layer {
}
export class SimpleRNNCell extends RNNCell {
    constructor(args) {
        super(args);
        this.DEFAULT_ACTIVATION = 'tanh';
        this.DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
        this.DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
        this.DEFAULT_BIAS_INITIALIZER = 'zeros';
        this.units = args.units;
        assertPositiveInteger(this.units, `units`);
        this.activation = getActivation(args.activation == null ? this.DEFAULT_ACTIVATION : args.activation);
        this.useBias = args.useBias == null ? true : args.useBias;
        this.kernelInitializer = getInitializer(args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
        this.recurrentInitializer = getInitializer(args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);
        this.biasInitializer =
            getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
        this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
        this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
        this.biasRegularizer = getRegularizer(args.biasRegularizer);
        this.kernelConstraint = getConstraint(args.kernelConstraint);
        this.recurrentConstraint = getConstraint(args.recurrentConstraint);
        this.biasConstraint = getConstraint(args.biasConstraint);
        this.dropout = math_utils.min([1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
        this.recurrentDropout = math_utils.min([
            1,
            math_utils.max([0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
        ]);
        this.dropoutFunc = args.dropoutFunc;
        this.stateSize = this.units;
        this.dropoutMask = null;
        this.recurrentDropoutMask = null;
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        // TODO(cais): Use regularizer.
        this.kernel = this.addWeight('kernel', [inputShape[inputShape.length - 1], this.units], null, this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
        this.recurrentKernel = this.addWeight('recurrent_kernel', [this.units, this.units], null, this.recurrentInitializer, this.recurrentRegularizer, true, this.recurrentConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [this.units], null, this.biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        else {
            this.bias = null;
        }
        this.built = true;
    }
    // Porting Note: PyKeras' equivalent of this method takes two tensor inputs:
    //   `inputs` and `states`. Here, the two tensors are combined into an
    //   `Tensor[]` Array as the first input argument.
    //   Similarly, PyKeras' equivalent of this method returns two values:
    //    `output` and `[output]`. Here the two are combined into one length-2
    //    `Tensor[]`, consisting of `output` repeated.
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = inputs;
            if (inputs.length !== 2) {
                throw new ValueError(`SimpleRNNCell expects 2 input Tensors, got ${inputs.length}.`);
            }
            let prevOutput = inputs[1];
            inputs = inputs[0];
            const training = kwargs['training'] == null ? false : kwargs['training'];
            if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
                this.dropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(inputs),
                    rate: this.dropout,
                    training,
                    dropoutFunc: this.dropoutFunc,
                });
            }
            if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
                this.recurrentDropoutMask == null) {
                this.recurrentDropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(prevOutput),
                    rate: this.recurrentDropout,
                    training,
                    dropoutFunc: this.dropoutFunc,
                });
            }
            let h;
            const dpMask = this.dropoutMask;
            const recDpMask = this.recurrentDropoutMask;
            if (dpMask != null) {
                h = K.dot(tfc.mul(inputs, dpMask), this.kernel.read());
            }
            else {
                h = K.dot(inputs, this.kernel.read());
            }
            if (this.bias != null) {
                h = K.biasAdd(h, this.bias.read());
            }
            if (recDpMask != null) {
                prevOutput = tfc.mul(prevOutput, recDpMask);
            }
            let output = tfc.add(h, K.dot(prevOutput, this.recurrentKernel.read()));
            if (this.activation != null) {
                output = this.activation.apply(output);
            }
            // TODO(cais): Properly set learning phase on output tensor?
            return [output, output];
        });
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = {
            units: this.units,
            activation: serializeActivation(this.activation),
            useBias: this.useBias,
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            recurrentInitializer: serializeInitializer(this.recurrentInitializer),
            biasInitializer: serializeInitializer(this.biasInitializer),
            kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
            recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
            biasRegularizer: serializeRegularizer(this.biasRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            kernelConstraint: serializeConstraint(this.kernelConstraint),
            recurrentConstraint: serializeConstraint(this.recurrentConstraint),
            biasConstraint: serializeConstraint(this.biasConstraint),
            dropout: this.dropout,
            recurrentDropout: this.recurrentDropout,
        };
        return Object.assign(Object.assign({}, baseConfig), config);
    }
}
/** @nocollapse */
SimpleRNNCell.className = 'SimpleRNNCell';
serialization.registerClass(SimpleRNNCell);
export class SimpleRNN extends RNN {
    constructor(args) {
        args.cell = new SimpleRNNCell(args);
        super(args);
        // TODO(cais): Add activityRegularizer.
    }
    call(inputs, kwargs) {
        return tidy(() => {
            if (this.cell.dropoutMask != null) {
                tfc.dispose(this.cell.dropoutMask);
                this.cell.dropoutMask = null;
            }
            if (this.cell.recurrentDropoutMask != null) {
                tfc.dispose(this.cell.recurrentDropoutMask);
                this.cell.recurrentDropoutMask = null;
            }
            const mask = kwargs == null ? null : kwargs['mask'];
            const training = kwargs == null ? null : kwargs['training'];
            const initialState = kwargs == null ? null : kwargs['initialState'];
            return super.call(inputs, { mask, training, initialState });
        });
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config);
    }
}
/** @nocollapse */
SimpleRNN.className = 'SimpleRNN';
serialization.registerClass(SimpleRNN);
export class GRUCell extends RNNCell {
    constructor(args) {
        super(args);
        this.DEFAULT_ACTIVATION = 'tanh';
        this.DEFAULT_RECURRENT_ACTIVATION = 'hardSigmoid';
        this.DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
        this.DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
        this.DEFAULT_BIAS_INITIALIZER = 'zeros';
        if (args.resetAfter) {
            throw new ValueError(`GRUCell does not support reset_after parameter set to true.`);
        }
        this.units = args.units;
        assertPositiveInteger(this.units, 'units');
        this.activation = getActivation(args.activation === undefined ? this.DEFAULT_ACTIVATION :
            args.activation);
        this.recurrentActivation = getActivation(args.recurrentActivation === undefined ?
            this.DEFAULT_RECURRENT_ACTIVATION :
            args.recurrentActivation);
        this.useBias = args.useBias == null ? true : args.useBias;
        this.kernelInitializer = getInitializer(args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
        this.recurrentInitializer = getInitializer(args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);
        this.biasInitializer =
            getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
        this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
        this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
        this.biasRegularizer = getRegularizer(args.biasRegularizer);
        this.kernelConstraint = getConstraint(args.kernelConstraint);
        this.recurrentConstraint = getConstraint(args.recurrentConstraint);
        this.biasConstraint = getConstraint(args.biasConstraint);
        this.dropout = math_utils.min([1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
        this.recurrentDropout = math_utils.min([
            1,
            math_utils.max([0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
        ]);
        this.dropoutFunc = args.dropoutFunc;
        this.implementation = args.implementation;
        this.stateSize = this.units;
        this.dropoutMask = null;
        this.recurrentDropoutMask = null;
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const inputDim = inputShape[inputShape.length - 1];
        this.kernel = this.addWeight('kernel', [inputDim, this.units * 3], null, this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
        this.recurrentKernel = this.addWeight('recurrent_kernel', [this.units, this.units * 3], null, this.recurrentInitializer, this.recurrentRegularizer, true, this.recurrentConstraint);
        if (this.useBias) {
            this.bias = this.addWeight('bias', [this.units * 3], null, this.biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        else {
            this.bias = null;
        }
        // Porting Notes: Unlike the PyKeras implementation, we perform slicing
        //   of the weights and bias in the call() method, at execution time.
        this.built = true;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = inputs;
            if (inputs.length !== 2) {
                throw new ValueError(`GRUCell expects 2 input Tensors (inputs, h, c), got ` +
                    `${inputs.length}.`);
            }
            const training = kwargs['training'] == null ? false : kwargs['training'];
            let hTMinus1 = inputs[1]; // Previous memory state.
            inputs = inputs[0];
            // Note: For superior performance, TensorFlow.js always uses
            // implementation 2, regardless of the actual value of
            // config.implementation.
            if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
                this.dropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(inputs),
                    rate: this.dropout,
                    training,
                    count: 3,
                    dropoutFunc: this.dropoutFunc,
                });
            }
            if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
                this.recurrentDropoutMask == null) {
                this.recurrentDropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(hTMinus1),
                    rate: this.recurrentDropout,
                    training,
                    count: 3,
                    dropoutFunc: this.dropoutFunc,
                });
            }
            const dpMask = this.dropoutMask;
            const recDpMask = this.recurrentDropoutMask;
            let z;
            let r;
            let hh;
            if (0 < this.dropout && this.dropout < 1) {
                inputs = tfc.mul(inputs, dpMask[0]);
            }
            let matrixX = K.dot(inputs, this.kernel.read());
            if (this.useBias) {
                matrixX = K.biasAdd(matrixX, this.bias.read());
            }
            if (0 < this.recurrentDropout && this.recurrentDropout < 1) {
                hTMinus1 = tfc.mul(hTMinus1, recDpMask[0]);
            }
            const recurrentKernelValue = this.recurrentKernel.read();
            const [rk1, rk2] = tfc.split(recurrentKernelValue, [2 * this.units, this.units], recurrentKernelValue.rank - 1);
            const matrixInner = K.dot(hTMinus1, rk1);
            const [xZ, xR, xH] = tfc.split(matrixX, 3, matrixX.rank - 1);
            const [recurrentZ, recurrentR] = tfc.split(matrixInner, 2, matrixInner.rank - 1);
            z = this.recurrentActivation.apply(tfc.add(xZ, recurrentZ));
            r = this.recurrentActivation.apply(tfc.add(xR, recurrentR));
            const recurrentH = K.dot(tfc.mul(r, hTMinus1), rk2);
            hh = this.activation.apply(tfc.add(xH, recurrentH));
            const h = tfc.add(tfc.mul(z, hTMinus1), tfc.mul(tfc.add(1, tfc.neg(z)), hh));
            // TODO(cais): Add use_learning_phase flag properly.
            return [h, h];
        });
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = {
            units: this.units,
            activation: serializeActivation(this.activation),
            recurrentActivation: serializeActivation(this.recurrentActivation),
            useBias: this.useBias,
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            recurrentInitializer: serializeInitializer(this.recurrentInitializer),
            biasInitializer: serializeInitializer(this.biasInitializer),
            kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
            recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
            biasRegularizer: serializeRegularizer(this.biasRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            kernelConstraint: serializeConstraint(this.kernelConstraint),
            recurrentConstraint: serializeConstraint(this.recurrentConstraint),
            biasConstraint: serializeConstraint(this.biasConstraint),
            dropout: this.dropout,
            recurrentDropout: this.recurrentDropout,
            implementation: this.implementation,
            resetAfter: false
        };
        return Object.assign(Object.assign({}, baseConfig), config);
    }
}
/** @nocollapse */
GRUCell.className = 'GRUCell';
serialization.registerClass(GRUCell);
export class GRU extends RNN {
    constructor(args) {
        if (args.implementation === 0) {
            console.warn('`implementation=0` has been deprecated, and now defaults to ' +
                '`implementation=1`. Please update your layer call.');
        }
        args.cell = new GRUCell(args);
        super(args);
        // TODO(cais): Add activityRegularizer.
    }
    call(inputs, kwargs) {
        return tidy(() => {
            if (this.cell.dropoutMask != null) {
                tfc.dispose(this.cell.dropoutMask);
                this.cell.dropoutMask = null;
            }
            if (this.cell.recurrentDropoutMask != null) {
                tfc.dispose(this.cell.recurrentDropoutMask);
                this.cell.recurrentDropoutMask = null;
            }
            const mask = kwargs == null ? null : kwargs['mask'];
            const training = kwargs == null ? null : kwargs['training'];
            const initialState = kwargs == null ? null : kwargs['initialState'];
            return super.call(inputs, { mask, training, initialState });
        });
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        if (config['implmentation'] === 0) {
            config['implementation'] = 1;
        }
        return new cls(config);
    }
}
/** @nocollapse */
GRU.className = 'GRU';
serialization.registerClass(GRU);
export class LSTMCell extends RNNCell {
    constructor(args) {
        super(args);
        this.DEFAULT_ACTIVATION = 'tanh';
        this.DEFAULT_RECURRENT_ACTIVATION = 'hardSigmoid';
        this.DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
        this.DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
        this.DEFAULT_BIAS_INITIALIZER = 'zeros';
        this.units = args.units;
        assertPositiveInteger(this.units, 'units');
        this.activation = getActivation(args.activation === undefined ? this.DEFAULT_ACTIVATION :
            args.activation);
        this.recurrentActivation = getActivation(args.recurrentActivation === undefined ?
            this.DEFAULT_RECURRENT_ACTIVATION :
            args.recurrentActivation);
        this.useBias = args.useBias == null ? true : args.useBias;
        this.kernelInitializer = getInitializer(args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
        this.recurrentInitializer = getInitializer(args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);
        this.biasInitializer =
            getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
        this.unitForgetBias = args.unitForgetBias;
        this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
        this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
        this.biasRegularizer = getRegularizer(args.biasRegularizer);
        this.kernelConstraint = getConstraint(args.kernelConstraint);
        this.recurrentConstraint = getConstraint(args.recurrentConstraint);
        this.biasConstraint = getConstraint(args.biasConstraint);
        this.dropout = math_utils.min([1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
        this.recurrentDropout = math_utils.min([
            1,
            math_utils.max([0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
        ]);
        this.dropoutFunc = args.dropoutFunc;
        this.implementation = args.implementation;
        this.stateSize = [this.units, this.units];
        this.dropoutMask = null;
        this.recurrentDropoutMask = null;
    }
    build(inputShape) {
        var _a;
        inputShape = getExactlyOneShape(inputShape);
        const inputDim = inputShape[inputShape.length - 1];
        this.kernel = this.addWeight('kernel', [inputDim, this.units * 4], null, this.kernelInitializer, this.kernelRegularizer, true, this.kernelConstraint);
        this.recurrentKernel = this.addWeight('recurrent_kernel', [this.units, this.units * 4], null, this.recurrentInitializer, this.recurrentRegularizer, true, this.recurrentConstraint);
        let biasInitializer;
        if (this.useBias) {
            if (this.unitForgetBias) {
                const capturedBiasInit = this.biasInitializer;
                const capturedUnits = this.units;
                biasInitializer = new (_a = class CustomInit extends Initializer {
                        apply(shape, dtype) {
                            // TODO(cais): More informative variable names?
                            const bI = capturedBiasInit.apply([capturedUnits]);
                            const bF = (new Ones()).apply([capturedUnits]);
                            const bCAndH = capturedBiasInit.apply([capturedUnits * 2]);
                            return K.concatAlongFirstAxis(K.concatAlongFirstAxis(bI, bF), bCAndH);
                        }
                    },
                    /** @nocollapse */
                    _a.className = 'CustomInit',
                    _a)();
            }
            else {
                biasInitializer = this.biasInitializer;
            }
            this.bias = this.addWeight('bias', [this.units * 4], null, biasInitializer, this.biasRegularizer, true, this.biasConstraint);
        }
        else {
            this.bias = null;
        }
        // Porting Notes: Unlike the PyKeras implementation, we perform slicing
        //   of the weights and bias in the call() method, at execution time.
        this.built = true;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            const training = kwargs['training'] == null ? false : kwargs['training'];
            inputs = inputs;
            if (inputs.length !== 3) {
                throw new ValueError(`LSTMCell expects 3 input Tensors (inputs, h, c), got ` +
                    `${inputs.length}.`);
            }
            let hTMinus1 = inputs[1]; // Previous memory state.
            const cTMinus1 = inputs[2]; // Previous carry state.
            inputs = inputs[0];
            if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
                this.dropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(inputs),
                    rate: this.dropout,
                    training,
                    count: 4,
                    dropoutFunc: this.dropoutFunc
                });
            }
            if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
                this.recurrentDropoutMask == null) {
                this.recurrentDropoutMask = generateDropoutMask({
                    ones: () => tfc.onesLike(hTMinus1),
                    rate: this.recurrentDropout,
                    training,
                    count: 4,
                    dropoutFunc: this.dropoutFunc
                });
            }
            const dpMask = this.dropoutMask;
            const recDpMask = this.recurrentDropoutMask;
            // Note: For superior performance, TensorFlow.js always uses
            // implementation 2 regardless of the actual value of
            // config.implementation.
            let i;
            let f;
            let c;
            let o;
            if (0 < this.dropout && this.dropout < 1) {
                inputs = tfc.mul(inputs, dpMask[0]);
            }
            let z = K.dot(inputs, this.kernel.read());
            if (0 < this.recurrentDropout && this.recurrentDropout < 1) {
                hTMinus1 = tfc.mul(hTMinus1, recDpMask[0]);
            }
            z = tfc.add(z, K.dot(hTMinus1, this.recurrentKernel.read()));
            if (this.useBias) {
                z = K.biasAdd(z, this.bias.read());
            }
            const [z0, z1, z2, z3] = tfc.split(z, 4, z.rank - 1);
            i = this.recurrentActivation.apply(z0);
            f = this.recurrentActivation.apply(z1);
            c = tfc.add(tfc.mul(f, cTMinus1), tfc.mul(i, this.activation.apply(z2)));
            o = this.recurrentActivation.apply(z3);
            const h = tfc.mul(o, this.activation.apply(c));
            // TODO(cais): Add use_learning_phase flag properly.
            return [h, h, c];
        });
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const config = {
            units: this.units,
            activation: serializeActivation(this.activation),
            recurrentActivation: serializeActivation(this.recurrentActivation),
            useBias: this.useBias,
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            recurrentInitializer: serializeInitializer(this.recurrentInitializer),
            biasInitializer: serializeInitializer(this.biasInitializer),
            unitForgetBias: this.unitForgetBias,
            kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
            recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
            biasRegularizer: serializeRegularizer(this.biasRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            kernelConstraint: serializeConstraint(this.kernelConstraint),
            recurrentConstraint: serializeConstraint(this.recurrentConstraint),
            biasConstraint: serializeConstraint(this.biasConstraint),
            dropout: this.dropout,
            recurrentDropout: this.recurrentDropout,
            implementation: this.implementation,
        };
        return Object.assign(Object.assign({}, baseConfig), config);
    }
}
/** @nocollapse */
LSTMCell.className = 'LSTMCell';
serialization.registerClass(LSTMCell);
export class LSTM extends RNN {
    constructor(args) {
        if (args.implementation === 0) {
            console.warn('`implementation=0` has been deprecated, and now defaults to ' +
                '`implementation=1`. Please update your layer call.');
        }
        args.cell = new LSTMCell(args);
        super(args);
        // TODO(cais): Add activityRegularizer.
    }
    call(inputs, kwargs) {
        return tidy(() => {
            if (this.cell.dropoutMask != null) {
                tfc.dispose(this.cell.dropoutMask);
                this.cell.dropoutMask = null;
            }
            if (this.cell.recurrentDropoutMask != null) {
                tfc.dispose(this.cell.recurrentDropoutMask);
                this.cell.recurrentDropoutMask = null;
            }
            const mask = kwargs == null ? null : kwargs['mask'];
            const training = kwargs == null ? null : kwargs['training'];
            const initialState = kwargs == null ? null : kwargs['initialState'];
            return super.call(inputs, { mask, training, initialState });
        });
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        if (config['implmentation'] === 0) {
            config['implementation'] = 1;
        }
        return new cls(config);
    }
}
/** @nocollapse */
LSTM.className = 'LSTM';
serialization.registerClass(LSTM);
export class StackedRNNCells extends RNNCell {
    constructor(args) {
        super(args);
        this.cells = args.cells;
    }
    get stateSize() {
        // States are a flat list in reverse order of the cell stack.
        // This allows perserving the requirement `stack.statesize[0] ===
        // outputDim`. E.g., states of a 2-layer LSTM would be `[h2, c2, h1, c1]`,
        // assuming one LSTM has states `[h, c]`.
        const stateSize = [];
        for (const cell of this.cells.slice().reverse()) {
            if (Array.isArray(cell.stateSize)) {
                stateSize.push(...cell.stateSize);
            }
            else {
                stateSize.push(cell.stateSize);
            }
        }
        return stateSize;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = inputs;
            let states = inputs.slice(1);
            // Recover per-cell states.
            const nestedStates = [];
            for (const cell of this.cells.slice().reverse()) {
                if (Array.isArray(cell.stateSize)) {
                    nestedStates.push(states.splice(0, cell.stateSize.length));
                }
                else {
                    nestedStates.push(states.splice(0, 1));
                }
            }
            nestedStates.reverse();
            // Call the cells in order and store the returned states.
            const newNestedStates = [];
            let callInputs;
            for (let i = 0; i < this.cells.length; ++i) {
                const cell = this.cells[i];
                states = nestedStates[i];
                // TODO(cais): Take care of constants.
                if (i === 0) {
                    callInputs = [inputs[0]].concat(states);
                }
                else {
                    callInputs = [callInputs[0]].concat(states);
                }
                callInputs = cell.call(callInputs, kwargs);
                newNestedStates.push(callInputs.slice(1));
            }
            // Format the new states as a flat list in reverse cell order.
            states = [];
            for (const cellStates of newNestedStates.slice().reverse()) {
                states.push(...cellStates);
            }
            return [callInputs[0]].concat(states);
        });
    }
    build(inputShape) {
        if (isArrayOfShapes(inputShape)) {
            // TODO(cais): Take care of input constants.
            // const constantShape = inputShape.slice(1);
            inputShape = inputShape[0];
        }
        inputShape = inputShape;
        let outputDim;
        this.cells.forEach((cell, i) => {
            nameScope(`RNNCell_${i}`, () => {
                // TODO(cais): Take care of input constants.
                cell.build(inputShape);
                if (Array.isArray(cell.stateSize)) {
                    outputDim = cell.stateSize[0];
                }
                else {
                    outputDim = cell.stateSize;
                }
                inputShape = [inputShape[0], outputDim];
            });
        });
        this.built = true;
    }
    getConfig() {
        const baseConfig = super.getConfig();
        const getCellConfig = (cell) => {
            return {
                'className': cell.getClassName(),
                'config': cell.getConfig(),
            };
        };
        const cellConfigs = this.cells.map(getCellConfig);
        const config = { 'cells': cellConfigs };
        return Object.assign(Object.assign({}, baseConfig), config);
    }
    /** @nocollapse */
    static fromConfig(cls, config, customObjects = {}) {
        const cells = [];
        for (const cellConfig of config['cells']) {
            cells.push(deserialize(cellConfig, customObjects));
        }
        return new cls({ cells });
    }
    get trainableWeights() {
        if (!this.trainable) {
            return [];
        }
        const weights = [];
        for (const cell of this.cells) {
            weights.push(...cell.trainableWeights);
        }
        return weights;
    }
    get nonTrainableWeights() {
        const weights = [];
        for (const cell of this.cells) {
            weights.push(...cell.nonTrainableWeights);
        }
        if (!this.trainable) {
            const trainableWeights = [];
            for (const cell of this.cells) {
                trainableWeights.push(...cell.trainableWeights);
            }
            return trainableWeights.concat(weights);
        }
        return weights;
    }
    /**
     * Retrieve the weights of a the model.
     *
     * @returns A flat `Array` of `tf.Tensor`s.
     */
    getWeights() {
        const weights = [];
        for (const cell of this.cells) {
            weights.push(...cell.weights);
        }
        return batchGetValue(weights);
    }
    /**
     * Set the weights of the model.
     *
     * @param weights An `Array` of `tf.Tensor`s with shapes and types matching
     *     the output of `getWeights()`.
     */
    setWeights(weights) {
        const tuples = [];
        for (const cell of this.cells) {
            const numParams = cell.weights.length;
            const inputWeights = weights.splice(numParams);
            for (let i = 0; i < cell.weights.length; ++i) {
                tuples.push([cell.weights[i], inputWeights[i]]);
            }
        }
        batchSetValue(tuples);
    }
}
/** @nocollapse */
StackedRNNCells.className = 'StackedRNNCells';
serialization.registerClass(StackedRNNCells);
export function generateDropoutMask(args) {
    const { ones, rate, training = false, count = 1, dropoutFunc } = args;
    const droppedInputs = () => dropoutFunc != null ? dropoutFunc(ones(), rate) : K.dropout(ones(), rate);
    const createMask = () => K.inTrainPhase(droppedInputs, ones, training);
    // just in case count is provided with null or undefined
    if (!count || count <= 1) {
        return tfc.keep(createMask().clone());
    }
    const masks = Array(count).fill(undefined).map(createMask);
    return masks.map(m => tfc.keep(m.clone()));
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVjdXJyZW50LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1sYXllcnMvc3JjL2xheWVycy9yZWN1cnJlbnQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0dBUUc7QUFFSDs7R0FFRztBQUVILE9BQU8sS0FBSyxHQUFHLE1BQU0sdUJBQXVCLENBQUM7QUFDN0MsT0FBTyxFQUFXLGFBQWEsRUFBVSxJQUFJLEVBQUUsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFbEYsT0FBTyxFQUFhLGFBQWEsRUFBRSxtQkFBbUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQzlFLE9BQU8sS0FBSyxDQUFDLE1BQU0seUJBQXlCLENBQUM7QUFDN0MsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNwQyxPQUFPLEVBQW1DLGFBQWEsRUFBRSxtQkFBbUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQ3BHLE9BQU8sRUFBQyxTQUFTLEVBQUUsY0FBYyxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDN0QsT0FBTyxFQUFDLEtBQUssRUFBWSxNQUFNLG9CQUFvQixDQUFDO0FBQ3BELE9BQU8sRUFBQyxjQUFjLEVBQUUsbUJBQW1CLEVBQUUsVUFBVSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQzFFLE9BQU8sRUFBQyxjQUFjLEVBQUUsV0FBVyxFQUF5QixJQUFJLEVBQUUsb0JBQW9CLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUcvRyxPQUFPLEVBQUMsY0FBYyxFQUFzQyxvQkFBb0IsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBRXpHLE9BQU8sRUFBQyxxQkFBcUIsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBQzdELE9BQU8sS0FBSyxVQUFVLE1BQU0scUJBQXFCLENBQUM7QUFDbEQsT0FBTyxFQUFDLGtCQUFrQixFQUFFLG1CQUFtQixFQUFFLGVBQWUsRUFBQyxNQUFNLHNCQUFzQixDQUFD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UUFBUSxDQUFDLENBQUM7SUFFdkUsd0RBQXdEO0lBQ3hELElBQUksQ0FBQyxLQUFLLElBQUksS0FBSyxJQUFJLENBQUMsRUFBRTtRQUN4QixPQUFPLEdBQUcsQ0FBQyxJQUFJLENBQUMsVUFBVSxFQUFFLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUN2QztJQUVELE1BQU0sS0FBSyxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUMsR0FBRyxDQUFDLFVBQVUsQ0FBQyxDQUFDO0lBRTNELE9BQU8sS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsQ0FBQztBQUM3QyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQ1xuICpcbiAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZVxuICogbGljZW5zZSB0aGF0IGNhbiBiZSBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIG9yIGF0XG4gKiBodHRwczovL29wZW5zb3VyY2Uub3JnL2xpY2Vuc2VzL01JVC5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyoqXG4gKiBUZW5zb3JGbG93LmpzIExheWVyczogUmVjdXJyZW50IE5ldXJhbCBOZXR3b3JrIExheWVycy5cbiAqL1xuXG5pbXBvcnQgKiBhcyB0ZmMgZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7RGF0YVR5cGUsIHNlcmlhbGl6YXRpb24sIFRlbnNvciwgdGlkeSwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtBY3RpdmF0aW9uLCBnZXRBY3RpdmF0aW9uLCBzZXJpYWxpemVBY3RpdmF0aW9ufSBmcm9tICcuLi9hY3RpdmF0aW9ucyc7XG5pbXBvcnQgKiBhcyBLIGZyb20gJy4uL2JhY2tlbmQvdGZqc19iYWNrZW5kJztcbmltcG9ydCB7bmFtZVNjb3BlfSBmcm9tICcuLi9jb21tb24nO1xuaW1wb3J0IHtDb25zdHJhaW50LCBDb25zdHJhaW50SWRlbnRpZmllciwgZ2V0Q29uc3RyYWludCwgc2VyaWFsaXplQ29uc3RyYWludH0gZnJvbSAnLi4vY29uc3RyYWludHMnO1xuaW1wb3J0IHtJbnB1dFNwZWMsIFN5bWJvbGljVGVuc29yfSBmcm9tICcuLi9lbmdpbmUvdG9wb2xvZ3knO1xuaW1wb3J0IHtMYXllciwgTGF5ZXJBcmdzfSBmcm9tICcuLi9lbmdpbmUvdG9wb2xvZ3knO1xuaW1wb3J0IHtBdHRyaWJ1dGVFcnJvciwgTm90SW1wbGVtZW50ZWRFcnJvciwgVmFsdWVFcnJvcn0gZnJvbSAnLi4vZXJyb3JzJztcbmltcG9ydCB7Z2V0SW5pdGlhbGl6ZXIsIEluaXRpYWxpemVyLCBJbml0aWFsaXplcklkZW50aWZpZXIsIE9uZXMsIHNlcmlhbGl6ZUluaXRpYWxpemVyfSBmcm9tICcuLi9pbml0aWFsaXplcnMnO1xuaW1wb3J0IHtBY3RpdmF0aW9uSWRlbnRpZmllcn0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L2FjdGl2YXRpb25fY29uZmlnJztcbmltcG9ydCB7U2hhcGV9IGZyb20gJy4uL2tlcmFzX2Zvcm1hdC9jb21tb24nO1xuaW1wb3J0IHtnZXRSZWd1bGFyaXplciwgUmVndWxhcml6ZXIsIFJlZ3VsYXJpemVySWRlbnRpZmllciwgc2VyaWFsaXplUmVndWxhcml6ZXJ9IGZyb20gJy4uL3JlZ3VsYXJpemVycyc7XG5pbXBvcnQge0t3YXJncywgUm5uU3RlcEZ1bmN0aW9ufSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2Fzc2VydFBvc2l0aXZlSW50ZWdlcn0gZnJvbSAnLi4vdXRpbHMvZ2VuZXJpY191dGlscyc7XG5pbXBvcnQgKiBhcyBtYXRoX3V0aWxzIGZyb20gJy4uL3V0aWxzL21hdGhfdXRpbHMnO1xuaW1wb3J0IHtnZXRFeGFjdGx5T25lU2hhcGUsIGdldEV4YWN0bHlPbmVUZW5zb3IsIGlzQXJyYXlPZlNoYXBlc30gZnJvbSAnLi4vdXRpbHMvdHlwZXNfdXRpbHMnO1xuaW1wb3J0IHtiYXRjaEdldFZhbHVlLCBiYXRjaFNldFZhbHVlLCBMYXllclZhcmlhYmxlfSBmcm9tICcuLi92YXJpYWJsZXMnO1xuXG5pbXBvcnQge2Rlc2VyaWFsaXplfSBmcm9tICcuL3NlcmlhbGl6YXRpb24nO1xuXG4vKipcbiAqIFN0YW5kYXJkaXplIGBhcHBseSgpYCBhcmdzIHRvIGEgc2luZ2xlIGxpc3Qgb2YgdGVuc29yIGlucHV0cy5cbiAqXG4gKiBXaGVuIHJ1bm5pbmcgYSBtb2RlbCBsb2FkZWQgZnJvbSBmaWxlLCB0aGUgaW5wdXQgdGVuc29ycyBgaW5pdGlhbFN0YXRlYCBhbmRcbiAqIGBjb25zdGFudHNgIGFyZSBwYXNzZWQgdG8gYFJOTi5hcHBseSgpYCBhcyBwYXJ0IG9mIGBpbnB1dHNgIGluc3RlYWQgb2YgdGhlXG4gKiBkZWRpY2F0ZWQga3dhcmdzIGZpZWxkcy4gYGlucHV0c2AgY29uc2lzdHMgb2ZcbiAqIGBbaW5wdXRzLCBpbml0aWFsU3RhdGUwLCBpbml0aWFsU3RhdGUxLCAuLi4sIGNvbnN0YW50MCwgY29uc3RhbnQxXWAgaW4gdGhpc1xuICogY2FzZS5cbiAqIFRoaXMgbWV0aG9kIG1ha2VzIHN1cmUgdGhhdCBhcmd1bWVudHMgYXJlXG4gKiBzZXBhcmF0ZWQgYW5kIHRoYXQgYGluaXRpYWxTdGF0ZWAgYW5kIGBjb25zdGFudHNgIGFyZSBgQXJyYXlgcyBvZiB0ZW5zb3JzXG4gKiAob3IgTm9uZSkuXG4gKlxuICogQHBhcmFtIGlucHV0cyBUZW5zb3Igb3IgYEFycmF5YCBvZiAgdGVuc29ycy5cbiAqIEBwYXJhbSBpbml0aWFsU3RhdGUgVGVuc29yIG9yIGBBcnJheWAgb2YgdGVuc29ycyBvciBgbnVsbGAvYHVuZGVmaW5lZGAuXG4gKiBAcGFyYW0gY29uc3RhbnRzIFRlbnNvciBvciBgQXJyYXlgIG9mIHRlbnNvcnMgb3IgYG51bGxgL2B1bmRlZmluZWRgLlxuICogQHJldHVybnMgQW4gb2JqZWN0IGNvbnNpc3Rpbmcgb2ZcbiAqICAgaW5wdXRzOiBBIHRlbnNvci5cbiAqICAgaW5pdGlhbFN0YXRlOiBgQXJyYXlgIG9mIHRlbnNvcnMgb3IgYG51bGxgLlxuICogICBjb25zdGFudHM6IGBBcnJheWAgb2YgdGVuc29ycyBvciBgbnVsbGAuXG4gKiBAdGhyb3dzIFZhbHVlRXJyb3IsIGlmIGBpbnB1dHNgIGlzIGFuIGBBcnJheWAgYnV0IGVpdGhlciBgaW5pdGlhbFN0YXRlYCBvclxuICogICBgY29uc3RhbnRzYCBpcyBwcm92aWRlZC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHN0YW5kYXJkaXplQXJncyhcbiAgICBpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcnxTeW1ib2xpY1RlbnNvcltdLFxuICAgIGluaXRpYWxTdGF0ZTogVGVuc29yfFRlbnNvcltdfFN5bWJvbGljVGVuc29yfFN5bWJvbGljVGVuc29yW10sXG4gICAgY29uc3RhbnRzOiBUZW5zb3J8VGVuc29yW118U3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXSxcbiAgICBudW1Db25zdGFudHM/OiBudW1iZXIpOiB7XG4gIGlucHV0czogVGVuc29yfFN5bWJvbGljVGVuc29yLFxuICBpbml0aWFsU3RhdGU6IFRlbnNvcltdfFN5bWJvbGljVGVuc29yW10sXG4gIGNvbnN0YW50czogVGVuc29yW118U3ltYm9saWNUZW5zb3JbXVxufSB7XG4gIGlmIChBcnJheS5pc0FycmF5KGlucHV0cykpIHtcbiAgICBpZiAoaW5pdGlhbFN0YXRlICE9IG51bGwgfHwgY29uc3RhbnRzICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdXaGVuIGlucHV0cyBpcyBhbiBhcnJheSwgbmVpdGhlciBpbml0aWFsU3RhdGUgb3IgY29uc3RhbnRzICcgK1xuICAgICAgICAgICdzaG91bGQgYmUgcHJvdmlkZWQnKTtcbiAgICB9XG4gICAgaWYgKG51bUNvbnN0YW50cyAhPSBudWxsKSB7XG4gICAgICBjb25zdGFudHMgPSBpbnB1dHMuc2xpY2UoaW5wdXRzLmxlbmd0aCAtIG51bUNvbnN0YW50cywgaW5wdXRzLmxlbmd0aCk7XG4gICAgICBpbnB1dHMgPSBpbnB1dHMuc2xpY2UoMCwgaW5wdXRzLmxlbmd0aCAtIG51bUNvbnN0YW50cyk7XG4gICAgfVxuICAgIGlmIChpbnB1dHMubGVuZ3RoID4gMSkge1xuICAgICAgaW5pdGlhbFN0YXRlID0gaW5wdXRzLnNsaWNlKDEsIGlucHV0cy5sZW5ndGgpO1xuICAgIH1cbiAgICBpbnB1dHMgPSBpbnB1dHNbMF07XG4gIH1cblxuICBmdW5jdGlvbiB0b0xpc3RPck51bGwoeDogVGVuc29yfFRlbnNvcltdfFN5bWJvbGljVGVuc29yfFxuICAgICAgICAgICAgICAgICAgICAgICAgU3ltYm9saWNUZW5zb3JbXSk6IFRlbnNvcltdfFN5bWJvbGljVGVuc29yW10ge1xuICAgIGlmICh4ID09IG51bGwgfHwgQXJyYXkuaXNBcnJheSh4KSkge1xuICAgICAgcmV0dXJuIHggYXMgVGVuc29yW10gfCBTeW1ib2xpY1RlbnNvcltdO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gW3hdIGFzIFRlbnNvcltdIHwgU3ltYm9saWNUZW5zb3JbXTtcbiAgICB9XG4gIH1cblxuICBpbml0aWFsU3RhdGUgPSB0b0xpc3RPck51bGwoaW5pdGlhbFN0YXRlKTtcbiAgY29uc3RhbnRzID0gdG9MaXN0T3JOdWxsKGNvbnN0YW50cyk7XG5cbiAgcmV0dXJuIHtpbnB1dHMsIGluaXRpYWxTdGF0ZSwgY29uc3RhbnRzfTtcbn1cblxuLyoqXG4gKiBJdGVyYXRlcyBvdmVyIHRoZSB0aW1lIGRpbWVuc2lvbiBvZiBhIHRlbnNvci5cbiAqXG4gKiBAcGFyYW0gc3RlcEZ1bmN0aW9uIFJOTiBzdGVwIGZ1bmN0aW9uLlxuICogICBQYXJhbWV0ZXJzOlxuICogICAgIGlucHV0czogdGVuc29yIHdpdGggc2hhcGUgYFtzYW1wbGVzLCAuLi5dYCAobm8gdGltZSBkaW1lbnNpb24pLFxuICogICAgICAgcmVwcmVzZW50aW5nIGlucHV0IGZvciB0aGUgYmF0Y2ggb2Ygc2FtcGxlcyBhdCBhIGNlcnRhaW4gdGltZSBzdGVwLlxuICogICAgIHN0YXRlczogYW4gQXJyYXkgb2YgdGVuc29ycy5cbiAqICAgUmV0dXJuczpcbiAqICAgICBvdXRwdXRzOiB0ZW5zb3Igd2l0aCBzaGFwZSBgW3NhbXBsZXMsIG91dHB1dERpbV1gIChubyB0aW1lIGRpbWVuc2lvbikuXG4gKiAgICAgbmV3U3RhdGVzOiBsaXN0IG9mIHRlbnNvcnMsIHNhbWUgbGVuZ3RoIGFuZCBzaGFwZXMgYXMgYHN0YXRlc2AuIFRoZSBmaXJzdFxuICogICAgICAgc3RhdGUgaW4gdGhlIGxpc3QgbXVzdCBiZSB0aGUgb3V0cHV0IHRlbnNvciBhdCB0aGUgcHJldmlvdXMgdGltZXN0ZXAuXG4gKiBAcGFyYW0gaW5wdXRzIFRlbnNvciBvZiB0ZW1wb3JhbCBkYXRhIG9mIHNoYXBlIGBbc2FtcGxlcywgdGltZSwgLi4uXWAgKGF0XG4gKiAgIGxlYXN0IDNEKS5cbiAqIEBwYXJhbSBpbml0aWFsU3RhdGVzIFRlbnNvciB3aXRoIHNoYXBlIGBbc2FtcGxlcywgb3V0cHV0RGltXWAgKG5vIHRpbWVcbiAqICAgZGltZW5zaW9uKSwgY29udGFpbmluZyB0aGUgaW5pdGlhbCB2YWx1ZXMgb2YgdGhlIHN0YXRlcyB1c2VkIGluIHRoZSBzdGVwXG4gKiAgIGZ1bmN0aW9uLlxuICogQHBhcmFtIGdvQmFja3dhcmRzIElmIGB0cnVlYCwgZG8gdGhlIGl0ZXJhdGlvbiBvdmVyIHRoZSB0aW1lIGRpbWVuc2lvbiBpblxuICogICByZXZlcnNlIG9yZGVyIGFuZCByZXR1cm4gdGhlIHJldmVyc2VkIHNlcXVlbmNlLlxuICogQHBhcmFtIG1hc2sgQmluYXJ5IHRlbnNvciB3aXRoIHNoYXBlIGBbc2FtcGxlLCB0aW1lLCAxXWAsIHdpdGggYSB6ZXJvIGZvclxuICogICBldmVyeSBlbGVtZW50IHRoYXQgaXMgbWFza2VkLlxuICogQHBhcmFtIGNvbnN0YW50cyBBbiBBcnJheSBvZiBjb25zdGFudCB2YWx1ZXMgcGFzc2VkIGF0IGVhY2ggc3RlcC5cbiAqIEBwYXJhbSB1bnJvbGwgV2hldGhlciB0byB1bnJvbGwgdGhlIFJOTiBvciB0byB1c2UgYSBzeW1ib2xpYyBsb29wLiAqTm90KlxuICogICBhcHBsaWNhYmxlIHRvIHRoaXMgaW1wZXJhdGl2ZSBkZWVwbGVhcm4uanMgYmFja2VuZC4gSXRzIHZhbHVlIGlzIGlnbm9yZWQuXG4gKiBAcGFyYW0gbmVlZFBlclN0ZXBPdXRwdXRzIFdoZXRoZXIgdGhlIHBlci1zdGVwIG91dHB1dHMgYXJlIHRvIGJlXG4gKiAgIGNvbmNhdGVuYXRlZCBpbnRvIGEgc2luZ2xlIHRlbnNvciBhbmQgcmV0dXJuZWQgKGFzIHRoZSBzZWNvbmQgcmV0dXJuXG4gKiAgIHZhbHVlKS4gRGVmYXVsdDogYGZhbHNlYC4gVGhpcyBhcmcgaXMgaW5jbHVkZWQgc28gdGhhdCB0aGUgcmVsYXRpdmVseVxuICogICBleHBlbnNpdmUgY29uY2F0ZW5hdGlvbiBvZiB0aGUgc3RlcHdpc2Ugb3V0cHV0cyBjYW4gYmUgb21pdHRlZCB1bmxlc3NcbiAqICAgdGhlIHN0ZXB3aXNlIG91dHB1dHMgbmVlZCB0byBiZSBrZXB0IChlLmcuLCBmb3IgYW4gTFNUTSBsYXllciBvZiB3aGljaFxuICogICBgcmV0dXJuU2VxdWVuY2VgIGlzIGB0cnVlYC4pXG4gKiBAcmV0dXJucyBBbiBBcnJheTogYFtsYXN0T3V0cHV0LCBvdXRwdXRzLCBuZXdTdGF0ZXNdYC5cbiAqICAgbGFzdE91dHB1dDogdGhlIGxhc3Rlc3Qgb3V0cHV0IG9mIHRoZSBSTk4sIG9mIHNoYXBlIGBbc2FtcGxlcywgLi4uXWAuXG4gKiAgIG91dHB1dHM6IHRlbnNvciB3aXRoIHNoYXBlIGBbc2FtcGxlcywgdGltZSwgLi4uXWAgd2hlcmUgZWFjaCBlbnRyeVxuICogICAgIGBvdXRwdXRbcywgdF1gIGlzIHRoZSBvdXRwdXQgb2YgdGhlIHN0ZXAgZnVuY3Rpb24gYXQgdGltZSBgdGAgZm9yIHNhbXBsZVxuICogICAgIGBzYC4gVGhpcyByZXR1cm4gdmFsdWUgaXMgcHJvdmlkZWQgaWYgYW5kIG9ubHkgaWYgdGhlXG4gKiAgICAgYG5lZWRQZXJTdGVwT3V0cHV0c2AgaXMgc2V0IGFzIGB0cnVlYC4gSWYgaXQgaXMgc2V0IGFzIGBmYWxzZWAsIHRoaXNcbiAqICAgICByZXR1cm4gdmFsdWUgd2lsbCBiZSBgdW5kZWZpbmVkYC5cbiAqICAgbmV3U3RhdGVzOiBBcnJheSBvZiB0ZW5zb3JzLCBsYXRlc3Qgc3RhdGVzIHJldHVybmVkIGJ5IHRoZSBzdGVwIGZ1bmN0aW9uLFxuICogICAgICBvZiBzaGFwZSBgKHNhbXBsZXMsIC4uLilgLlxuICogQHRocm93cyBWYWx1ZUVycm9yIElmIGlucHV0IGRpbWVuc2lvbiBpcyBsZXNzIHRoYW4gMy5cbiAqXG4gKiBUT0RPKG5pZWxzZW5lKTogVGhpcyBuZWVkcyB0byBiZSB0aWR5LWVkLlxuICovXG5leHBvcnQgZnVuY3Rpb24gcm5uKFxuICAgIHN0ZXBGdW5jdGlvbjogUm5uU3RlcEZ1bmN0aW9uLCBpbnB1dHM6IFRlbnNvciwgaW5pdGlhbFN0YXRlczogVGVuc29yW10sXG4gICAgZ29CYWNrd2FyZHMgPSBmYWxzZSwgbWFzaz86IFRlbnNvciwgY29uc3RhbnRzPzogVGVuc29yW10sIHVucm9sbCA9IGZhbHNlLFxuICAgIG5lZWRQZXJTdGVwT3V0cHV0cyA9IGZhbHNlKTogW1RlbnNvciwgVGVuc29yLCBUZW5zb3JbXV0ge1xuICByZXR1cm4gdGZjLnRpZHkoKCkgPT4ge1xuICAgIGNvbnN0IG5kaW0gPSBpbnB1dHMuc2hhcGUubGVuZ3RoO1xuICAgIGlmIChuZGltIDwgMykge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoYElucHV0IHNob3VsZCBiZSBhdCBsZWFzdCAzRCwgYnV0IGlzICR7bmRpbX1ELmApO1xuICAgIH1cblxuICAgIC8vIFRyYW5zcG9zZSB0byB0aW1lLW1ham9yLCBpLmUuLCBmcm9tIFtiYXRjaCwgdGltZSwgLi4uXSB0byBbdGltZSwgYmF0Y2gsXG4gICAgLy8gLi4uXS5cbiAgICBjb25zdCBheGVzID0gWzEsIDBdLmNvbmNhdChtYXRoX3V0aWxzLnJhbmdlKDIsIG5kaW0pKTtcbiAgICBpbnB1dHMgPSB0ZmMudHJhbnNwb3NlKGlucHV0cywgYXhlcyk7XG5cbiAgICBpZiAoY29uc3RhbnRzICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgICdUaGUgcm5uKCkgZnVuY3RvaW4gb2YgdGhlIGRlZXBsZWFybi5qcyBiYWNrZW5kIGRvZXMgbm90IHN1cHBvcnQgJyArXG4gICAgICAgICAgJ2NvbnN0YW50cyB5ZXQuJyk7XG4gICAgfVxuXG4gICAgLy8gUG9ydGluZyBOb3RlOiB0aGUgdW5yb2xsIG9wdGlvbiBpcyBpZ25vcmVkIGJ5IHRoZSBpbXBlcmF0aXZlIGJhY2tlbmQuXG4gICAgaWYgKHVucm9sbCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICdCYWNrZW5kIHJubigpOiB0aGUgdW5yb2xsID0gdHJ1ZSBvcHRpb24gaXMgbm90IGFwcGxpY2FibGUgdG8gdGhlICcgK1xuICAgICAgICAgICdpbXBlcmF0aXZlIGRlZXBsZWFybi5qcyBiYWNrZW5kLicpO1xuICAgIH1cblxuICAgIGlmIChtYXNrICE9IG51bGwpIHtcbiAgICAgIG1hc2sgPSB0ZmMuY2FzdCh0ZmMuY2FzdChtYXNrLCAnYm9vbCcpLCAnZmxvYXQzMicpO1xuICAgICAgaWYgKG1hc2sucmFuayA9PT0gbmRpbSAtIDEpIHtcbiAgICAgICAgbWFzayA9IHRmYy5leHBhbmREaW1zKG1hc2ssIC0xKTtcbiAgICAgIH1cbiAgICAgIG1hc2sgPSB0ZmMudHJhbnNwb3NlKG1hc2ssIGF4ZXMpO1xuICAgIH1cblxuICAgIGlmIChnb0JhY2t3YXJkcykge1xuICAgICAgaW5wdXRzID0gdGZjLnJldmVyc2UoaW5wdXRzLCAwKTtcbiAgICAgIGlmIChtYXNrICE9IG51bGwpIHtcbiAgICAgICAgbWFzayA9IHRmYy5yZXZlcnNlKG1hc2ssIDApO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIFBvcnRpbmcgTm90ZTogUHlLZXJhcyB3aXRoIFRlbnNvckZsb3cgYmFja2VuZCB1c2VzIGEgc3ltYm9saWMgbG9vcFxuICAgIC8vICAgKHRmLndoaWxlX2xvb3ApLiBCdXQgZm9yIHRoZSBpbXBlcmF0aXZlIGRlZXBsZWFybi5qcyBiYWNrZW5kLCB3ZSBqdXN0XG4gICAgLy8gICB1c2UgdGhlIHVzdWFsIFR5cGVTY3JpcHQgY29udHJvbCBmbG93IHRvIGl0ZXJhdGUgb3ZlciB0aGUgdGltZSBzdGVwcyBpblxuICAgIC8vICAgdGhlIGlucHV0cy5cbiAgICAvLyBQb3J0aW5nIE5vdGU6IFB5S2VyYXMgcGF0Y2hlcyBhIFwiX3VzZV9sZWFybmluZ19waGFzZVwiIGF0dHJpYnV0ZSB0b1xuICAgIC8vIG91dHB1dHMuXG4gICAgLy8gICBUaGlzIGlzIG5vdCBpZGlvbWF0aWMgaW4gVHlwZVNjcmlwdC4gVGhlIGluZm8gcmVnYXJkaW5nIHdoZXRoZXIgd2UgYXJlXG4gICAgLy8gICBpbiBhIGxlYXJuaW5nIChpLmUuLCB0cmFpbmluZykgcGhhc2UgZm9yIFJOTiBpcyBwYXNzZWQgaW4gYSBkaWZmZXJlbnRcbiAgICAvLyAgIHdheS5cblxuICAgIGNvbnN0IHBlclN0ZXBPdXRwdXRzOiBUZW5zb3JbXSA9IFtdO1xuICAgIGxldCBsYXN0T3V0cHV0OiBUZW5zb3I7XG4gICAgbGV0IHN0YXRlcyA9IGluaXRpYWxTdGF0ZXM7XG4gICAgY29uc3QgdGltZVN0ZXBzID0gaW5wdXRzLnNoYXBlWzBdO1xuICAgIGNvbnN0IHBlclN0ZXBJbnB1dHMgPSB0ZmMudW5zdGFjayhpbnB1dHMpO1xuICAgIGxldCBwZXJTdGVwTWFza3M6IFRlbnNvcltdO1xuICAgIGlmIChtYXNrICE9IG51bGwpIHtcbiAgICAgIHBlclN0ZXBNYXNrcyA9IHRmYy51bnN0YWNrKG1hc2spO1xuICAgIH1cblxuICAgIGZvciAobGV0IHQgPSAwOyB0IDwgdGltZVN0ZXBzOyArK3QpIHtcbiAgICAgIGNvbnN0IGN1cnJlbnRJbnB1dCA9IHBlclN0ZXBJbnB1dHNbdF07XG4gICAgICBjb25zdCBzdGVwT3V0cHV0cyA9IHRmYy50aWR5KCgpID0+IHN0ZXBGdW5jdGlvbihjdXJyZW50SW5wdXQsIHN0YXRlcykpO1xuXG4gICAgICBpZiAobWFzayA9PSBudWxsKSB7XG4gICAgICAgIGxhc3RPdXRwdXQgPSBzdGVwT3V0cHV0c1swXTtcbiAgICAgICAgc3RhdGVzID0gc3RlcE91dHB1dHNbMV07XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBjb25zdCBtYXNrZWRPdXRwdXRzID0gdGZjLnRpZHkoKCkgPT4ge1xuICAgICAgICAgIGNvbnN0IHN0ZXBNYXNrID0gcGVyU3RlcE1hc2tzW3RdO1xuICAgICAgICAgIGNvbnN0IG5lZ1N0ZXBNYXNrID0gdGZjLnN1Yih0ZmMub25lc0xpa2Uoc3RlcE1hc2spLCBzdGVwTWFzayk7XG4gICAgICAgICAgLy8gVE9ETyhjYWlzKTogV291bGQgdGZjLndoZXJlKCkgYmUgYmV0dGVyIGZvciBwZXJmb3JtYW5jZT9cbiAgICAgICAgICBjb25zdCBvdXRwdXQgPSB0ZmMuYWRkKFxuICAgICAgICAgICAgICB0ZmMubXVsKHN0ZXBPdXRwdXRzWzBdLCBzdGVwTWFzayksXG4gICAgICAgICAgICAgIHRmYy5tdWwoc3RhdGVzWzBdLCBuZWdTdGVwTWFzaykpO1xuICAgICAgICAgIGNvbnN0IG5ld1N0YXRlcyA9IHN0YXRlcy5tYXAoKHN0YXRlLCBpKSA9PiB7XG4gICAgICAgICAgICByZXR1cm4gdGZjLmFkZChcbiAgICAgICAgICAgICAgICB0ZmMubXVsKHN0ZXBPdXRwdXRzWzFdW2ldLCBzdGVwTWFzayksXG4gICAgICAgICAgICAgICAgdGZjLm11bChzdGF0ZSwgbmVnU3RlcE1hc2spKTtcbiAgICAgICAgICB9KTtcbiAgICAgICAgICByZXR1cm4ge291dHB1dCwgbmV3U3RhdGVzfTtcbiAgICAgICAgfSk7XG4gICAgICAgIGxhc3RPdXRwdXQgPSBtYXNrZWRPdXRwdXRzLm91dHB1dDtcbiAgICAgICAgc3RhdGVzID0gbWFza2VkT3V0cHV0cy5uZXdTdGF0ZXM7XG4gICAgICB9XG5cbiAgICAgIGlmIChuZWVkUGVyU3RlcE91dHB1dHMpIHtcbiAgICAgICAgcGVyU3RlcE91dHB1dHMucHVzaChsYXN0T3V0cHV0KTtcbiAgICAgIH1cbiAgICB9XG4gICAgbGV0IG91dHB1dHM6IFRlbnNvcjtcbiAgICBpZiAobmVlZFBlclN0ZXBPdXRwdXRzKSB7XG4gICAgICBjb25zdCBheGlzID0gMTtcbiAgICAgIG91dHB1dHMgPSB0ZmMuc3RhY2socGVyU3RlcE91dHB1dHMsIGF4aXMpO1xuICAgIH1cbiAgICByZXR1cm4gW2xhc3RPdXRwdXQsIG91dHB1dHMsIHN0YXRlc10gYXMgW1RlbnNvciwgVGVuc29yLCBUZW5zb3JbXV07XG4gIH0pO1xufVxuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgQmFzZVJOTkxheWVyQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIC8qKlxuICAgKiBBIFJOTiBjZWxsIGluc3RhbmNlLiBBIFJOTiBjZWxsIGlzIGEgY2xhc3MgdGhhdCBoYXM6XG4gICAqICAgLSBhIGBjYWxsKClgIG1ldGhvZCwgd2hpY2ggdGFrZXMgYFtUZW5zb3IsIFRlbnNvcl1gIGFzIHRoZVxuICAgKiAgICAgZmlyc3QgaW5wdXQgYXJndW1lbnQuIFRoZSBmaXJzdCBpdGVtIGlzIHRoZSBpbnB1dCBhdCB0aW1lIHQsIGFuZFxuICAgKiAgICAgc2Vjb25kIGl0ZW0gaXMgdGhlIGNlbGwgc3RhdGUgYXQgdGltZSB0LlxuICAgKiAgICAgVGhlIGBjYWxsKClgIG1ldGhvZCByZXR1cm5zIGBbb3V0cHV0QXRULCBzdGF0ZXNBdFRQbHVzMV1gLlxuICAgKiAgICAgVGhlIGBjYWxsKClgIG1ldGhvZCBvZiB0aGUgY2VsbCBjYW4gYWxzbyB0YWtlIHRoZSBhcmd1bWVudCBgY29uc3RhbnRzYCxcbiAgICogICAgIHNlZSBzZWN0aW9uIFwiTm90ZSBvbiBwYXNzaW5nIGV4dGVybmFsIGNvbnN0YW50c1wiIGJlbG93LlxuICAgKiAgICAgUG9ydGluZyBOb2RlOiBQeUtlcmFzIG92ZXJyaWRlcyB0aGUgYGNhbGwoKWAgc2lnbmF0dXJlIG9mIFJOTiBjZWxscyxcbiAgICogICAgICAgd2hpY2ggYXJlIExheWVyIHN1YnR5cGVzLCB0byBhY2NlcHQgdHdvIGFyZ3VtZW50cy4gdGZqcy1sYXllcnMgZG9lc1xuICAgKiAgICAgICBub3QgZG8gc3VjaCBvdmVycmlkaW5nLiBJbnN0ZWFkIHdlIHByZXNldmUgdGhlIGBjYWxsKClgIHNpZ25hdHVyZSxcbiAgICogICAgICAgd2hpY2ggZHVlIHRvIGl0cyBgVGVuc29yfFRlbnNvcltdYCBhcmd1bWVudCBhbmQgcmV0dXJuIHZhbHVlIGlzXG4gICAqICAgICAgIGZsZXhpYmxlIGVub3VnaCB0byBoYW5kbGUgdGhlIGlucHV0cyBhbmQgc3RhdGVzLlxuICAgKiAgIC0gYSBgc3RhdGVTaXplYCBhdHRyaWJ1dGUuIFRoaXMgY2FuIGJlIGEgc2luZ2xlIGludGVnZXIgKHNpbmdsZSBzdGF0ZSlcbiAgICogICAgIGluIHdoaWNoIGNhc2UgaXQgaXMgdGhlIHNpemUgb2YgdGhlIHJlY3VycmVudCBzdGF0ZSAod2hpY2ggc2hvdWxkIGJlXG4gICAqICAgICB0aGUgc2FtZSBhcyB0aGUgc2l6ZSBvZiB0aGUgY2VsbCBvdXRwdXQpLiBUaGlzIGNhbiBhbHNvIGJlIGFuIEFycmF5IG9mXG4gICAqICAgICBpbnRlZ2VycyAob25lIHNpemUgcGVyIHN0YXRlKS4gSW4gdGhpcyBjYXNlLCB0aGUgZmlyc3QgZW50cnlcbiAgICogICAgIChgc3RhdGVTaXplWzBdYCkgc2hvdWxkIGJlIHRoZSBzYW1lIGFzIHRoZSBzaXplIG9mIHRoZSBjZWxsIG91dHB1dC5cbiAgICogSXQgaXMgYWxzbyBwb3NzaWJsZSBmb3IgYGNlbGxgIHRvIGJlIGEgbGlzdCBvZiBSTk4gY2VsbCBpbnN0YW5jZXMsIGluIHdoaWNoXG4gICAqIGNhc2UgdGhlIGNlbGxzIGdldCBzdGFja2VkIG9uIGFmdGVyIHRoZSBvdGhlciBpbiB0aGUgUk5OLCBpbXBsZW1lbnRpbmcgYW5cbiAgICogZWZmaWNpZW50IHN0YWNrZWQgUk5OLlxuICAgKi9cbiAgY2VsbD86IFJOTkNlbGx8Uk5OQ2VsbFtdO1xuXG4gIC8qKlxuICAgKiBXaGV0aGVyIHRvIHJldHVybiB0aGUgbGFzdCBvdXRwdXQgaW4gdGhlIG91dHB1dCBzZXF1ZW5jZSwgb3IgdGhlIGZ1bGxcbiAgICogc2VxdWVuY2UuXG4gICAqL1xuICByZXR1cm5TZXF1ZW5jZXM/OiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBXaGV0aGVyIHRvIHJldHVybiB0aGUgbGFzdCBzdGF0ZSBpbiBhZGRpdGlvbiB0byB0aGUgb3V0cHV0LlxuICAgKi9cbiAgcmV0dXJuU3RhdGU/OiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBJZiBgdHJ1ZWAsIHByb2Nlc3MgdGhlIGlucHV0IHNlcXVlbmNlIGJhY2t3YXJkcyBhbmQgcmV0dXJuIHRoZSByZXZlcnNlZFxuICAgKiBzZXF1ZW5jZSAoZGVmYXVsdDogYGZhbHNlYCkuXG4gICAqL1xuICBnb0JhY2t3YXJkcz86IGJvb2xlYW47XG5cbiAgLyoqXG4gICAqIElmIGB0cnVlYCwgdGhlIGxhc3Qgc3RhdGUgZm9yIGVhY2ggc2FtcGxlIGF0IGluZGV4IGkgaW4gYSBiYXRjaCB3aWxsIGJlXG4gICAqIHVzZWQgYXMgaW5pdGlhbCBzdGF0ZSBvZiB0aGUgc2FtcGxlIG9mIGluZGV4IGkgaW4gdGhlIGZvbGxvd2luZyBiYXRjaFxuICAgKiAoZGVmYXVsdDogYGZhbHNlYCkuXG4gICAqXG4gICAqIFlvdSBjYW4gc2V0IFJOTiBsYXllcnMgdG8gYmUgXCJzdGF0ZWZ1bFwiLCB3aGljaCBtZWFucyB0aGF0IHRoZSBzdGF0ZXNcbiAgICogY29tcHV0ZWQgZm9yIHRoZSBzYW1wbGVzIGluIG9uZSBiYXRjaCB3aWxsIGJlIHJldXNlZCBhcyBpbml0aWFsIHN0YXRlc1xuICAgKiBmb3IgdGhlIHNhbXBsZXMgaW4gdGhlIG5leHQgYmF0Y2guIFRoaXMgYXNzdW1lcyBhIG9uZS10by1vbmUgbWFwcGluZ1xuICAgKiBiZXR3ZWVuIHNhbXBsZXMgaW4gZGlmZmVyZW50IHN1Y2Nlc3NpdmUgYmF0Y2hlcy5cbiAgICpcbiAgICogVG8gZW5hYmxlIFwic3RhdGVmdWxuZXNzXCI6XG4gICAqICAgLSBzcGVjaWZ5IGBzdGF0ZWZ1bDogdHJ1ZWAgaW4gdGhlIGxheWVyIGNvbnN0cnVjdG9yLlxuICAgKiAgIC0gc3BlY2lmeSBhIGZpeGVkIGJhdGNoIHNpemUgZm9yIHlvdXIgbW9kZWwsIGJ5IHBhc3NpbmdcbiAgICogICAgIC0gaWYgc2VxdWVudGlhbCBtb2RlbDpcbiAgICogICAgICAgYGJhdGNoSW5wdXRTaGFwZTogWy4uLl1gIHRvIHRoZSBmaXJzdCBsYXllciBpbiB5b3VyIG1vZGVsLlxuICAgKiAgICAgLSBlbHNlIGZvciBmdW5jdGlvbmFsIG1vZGVsIHdpdGggMSBvciBtb3JlIElucHV0IGxheWVyczpcbiAgICogICAgICAgYGJhdGNoU2hhcGU6IFsuLi5dYCB0byBhbGwgdGhlIGZpcnN0IGxheWVycyBpbiB5b3VyIG1vZGVsLlxuICAgKiAgICAgVGhpcyBpcyB0aGUgZXhwZWN0ZWQgc2hhcGUgb2YgeW91ciBpbnB1dHNcbiAgICogICAgICppbmNsdWRpbmcgdGhlIGJhdGNoIHNpemUqLlxuICAgKiAgICAgSXQgc2hvdWxkIGJlIGEgdHVwbGUgb2YgaW50ZWdlcnMsIGUuZy4sIGBbMzIsIDEwLCAxMDBdYC5cbiAgICogICAtIHNwZWNpZnkgYHNodWZmbGU6IGZhbHNlYCB3aGVuIGNhbGxpbmcgYExheWVyc01vZGVsLmZpdCgpYC5cbiAgICpcbiAgICogVG8gcmVzZXQgdGhlIHN0YXRlIG9mIHlvdXIgbW9kZWwsIGNhbGwgYHJlc2V0U3RhdGVzKClgIG9uIGVpdGhlciB0aGVcbiAgICogc3BlY2lmaWMgbGF5ZXIgb3Igb24gdGhlIGVudGlyZSBtb2RlbC5cbiAgICovXG4gIHN0YXRlZnVsPzogYm9vbGVhbjtcbiAgLy8gVE9ETyhjYWlzKTogRXhwbG9yZSB3aGV0aGVyIHdlIGNhbiB3YXJuIHVzZXJzIHdoZW4gdGhleSBmYWlsIHRvIHNldFxuICAvLyAgIGBzaHVmZmxlOiBmYWxzZWAgd2hlbiB0cmFpbmluZyBhIG1vZGVsIGNvbnNpc3Rpbmcgb2Ygc3RhdGVmdWwgUk5Oc1xuICAvLyAgIGFuZCBhbnkgc3RhdGVmdWwgTGF5ZXJzIGluIGdlbmVyYWwuXG5cbiAgLyoqXG4gICAqIElmIGB0cnVlYCwgdGhlIG5ldHdvcmsgd2lsbCBiZSB1bnJvbGxlZCwgZWxzZSBhIHN5bWJvbGljIGxvb3Agd2lsbCBiZVxuICAgKiB1c2VkLiBVbnJvbGxpbmcgY2FuIHNwZWVkIHVwIGEgUk5OLCBhbHRob3VnaCBpdCB0ZW5kcyB0byBiZSBtb3JlXG4gICAqIG1lbW9yeS1pbnRlbnNpdmUuIFVucm9sbGluZyBpcyBvbmx5IHN1aXRhYmxlIGZvciBzaG9ydCBzZXF1ZW5jZXMgKGRlZmF1bHQ6XG4gICAqIGBmYWxzZWApLlxuICAgKiBQb3J0aW5nIE5vdGU6IHRmanMtbGF5ZXJzIGhhcyBhbiBpbXBlcmF0aXZlIGJhY2tlbmQuIFJOTnMgYXJlIGV4ZWN1dGVkIHdpdGhcbiAgICogICBub3JtYWwgVHlwZVNjcmlwdCBjb250cm9sIGZsb3cuIEhlbmNlIHRoaXMgcHJvcGVydHkgaXMgaW5hcHBsaWNhYmxlIGFuZFxuICAgKiAgIGlnbm9yZWQgaW4gdGZqcy1sYXllcnMuXG4gICAqL1xuICB1bnJvbGw/OiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBEaW1lbnNpb25hbGl0eSBvZiB0aGUgaW5wdXQgKGludGVnZXIpLlxuICAgKiAgIFRoaXMgb3B0aW9uIChvciBhbHRlcm5hdGl2ZWx5LCB0aGUgb3B0aW9uIGBpbnB1dFNoYXBlYCkgaXMgcmVxdWlyZWQgd2hlblxuICAgKiAgIHRoaXMgbGF5ZXIgaXMgdXNlZCBhcyB0aGUgZmlyc3QgbGF5ZXIgaW4gYSBtb2RlbC5cbiAgICovXG4gIGlucHV0RGltPzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBMZW5ndGggb2YgdGhlIGlucHV0IHNlcXVlbmNlcywgdG8gYmUgc3BlY2lmaWVkIHdoZW4gaXQgaXMgY29uc3RhbnQuXG4gICAqIFRoaXMgYXJndW1lbnQgaXMgcmVxdWlyZWQgaWYgeW91IGFyZSBnb2luZyB0byBjb25uZWN0IGBGbGF0dGVuYCB0aGVuXG4gICAqIGBEZW5zZWAgbGF5ZXJzIHVwc3RyZWFtICh3aXRob3V0IGl0LCB0aGUgc2hhcGUgb2YgdGhlIGRlbnNlIG91dHB1dHMgY2Fubm90XG4gICAqIGJlIGNvbXB1dGVkKS4gTm90ZSB0aGF0IGlmIHRoZSByZWN1cnJlbnQgbGF5ZXIgaXMgbm90IHRoZSBmaXJzdCBsYXllciBpblxuICAgKiB5b3VyIG1vZGVsLCB5b3Ugd291bGQgbmVlZCB0byBzcGVjaWZ5IHRoZSBpbnB1dCBsZW5ndGggYXQgdGhlIGxldmVsIG9mIHRoZVxuICAgKiBmaXJzdCBsYXllciAoZS5nLiwgdmlhIHRoZSBgaW5wdXRTaGFwZWAgb3B0aW9uKS5cbiAgICovXG4gIGlucHV0TGVuZ3RoPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgUk5OIGV4dGVuZHMgTGF5ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdSTk4nO1xuICBwdWJsaWMgcmVhZG9ubHkgY2VsbDogUk5OQ2VsbDtcbiAgcHVibGljIHJlYWRvbmx5IHJldHVyblNlcXVlbmNlczogYm9vbGVhbjtcbiAgcHVibGljIHJlYWRvbmx5IHJldHVyblN0YXRlOiBib29sZWFuO1xuICBwdWJsaWMgcmVhZG9ubHkgZ29CYWNrd2FyZHM6IGJvb2xlYW47XG4gIHB1YmxpYyByZWFkb25seSB1bnJvbGw6IGJvb2xlYW47XG5cbiAgcHVibGljIHN0YXRlU3BlYzogSW5wdXRTcGVjW107XG4gIHByb3RlY3RlZCBzdGF0ZXNfOiBUZW5zb3JbXTtcblxuICAvLyBOT1RFKGNhaXMpOiBGb3Igc3RhdGVmdWwgUk5OcywgdGhlIG9sZCBzdGF0ZXMgY2Fubm90IGJlIGRpc3Bvc2VkIHJpZ2h0XG4gIC8vIGF3YXkgd2hlbiBuZXcgc3RhdGVzIGFyZSBzZXQsIGJlY2F1c2UgdGhlIG9sZCBzdGF0ZXMgbWF5IG5lZWQgdG8gYmUgdXNlZFxuICAvLyBsYXRlciBmb3IgYmFja3Byb3BhZ2F0aW9uIHRocm91Z2ggdGltZSAoQlBUVCkgYW5kIG90aGVyIHB1cnBvc2VzLiBTbyB3ZVxuICAvLyBrZWVwIHRoZW0gaGVyZSBmb3IgZmluYWwgZGlzcG9zYWwgd2hlbiB0aGUgc3RhdGUgaXMgcmVzZXQgY29tcGxldGVseVxuICAvLyAoaS5lLiwgdGhyb3VnaCBuby1hcmcgY2FsbCB0byBgcmVzZXRTdGF0ZXMoKWApLlxuICBwcm90ZWN0ZWQga2VwdFN0YXRlczogVGVuc29yW11bXTtcblxuICBwcml2YXRlIG51bUNvbnN0YW50czogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFJOTkxheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIGxldCBjZWxsOiBSTk5DZWxsO1xuICAgIGlmIChhcmdzLmNlbGwgPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ2NlbGwgcHJvcGVydHkgaXMgbWlzc2luZyBmb3IgdGhlIGNvbnN0cnVjdG9yIG9mIFJOTi4nKTtcbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkoYXJncy5jZWxsKSkge1xuICAgICAgY2VsbCA9IG5ldyBTdGFja2VkUk5OQ2VsbHMoe2NlbGxzOiBhcmdzLmNlbGx9KTtcbiAgICB9IGVsc2Uge1xuICAgICAgY2VsbCA9IGFyZ3MuY2VsbDtcbiAgICB9XG4gICAgaWYgKGNlbGwuc3RhdGVTaXplID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdUaGUgUk5OIGNlbGwgc2hvdWxkIGhhdmUgYW4gYXR0cmlidXRlIGBzdGF0ZVNpemVgICh0dXBsZSBvZiAnICtcbiAgICAgICAgICAnaW50ZWdlcnMsIG9uZSBpbnRlZ2VyIHBlciBSTk4gc3RhdGUpLicpO1xuICAgIH1cbiAgICB0aGlzLmNlbGwgPSBjZWxsO1xuICAgIHRoaXMucmV0dXJuU2VxdWVuY2VzID1cbiAgICAgICAgYXJncy5yZXR1cm5TZXF1ZW5jZXMgPT0gbnVsbCA/IGZhbHNlIDogYXJncy5yZXR1cm5TZXF1ZW5jZXM7XG4gICAgdGhpcy5yZXR1cm5TdGF0ZSA9IGFyZ3MucmV0dXJuU3RhdGUgPT0gbnVsbCA/IGZhbHNlIDogYXJncy5yZXR1cm5TdGF0ZTtcbiAgICB0aGlzLmdvQmFja3dhcmRzID0gYXJncy5nb0JhY2t3YXJkcyA9PSBudWxsID8gZmFsc2UgOiBhcmdzLmdvQmFja3dhcmRzO1xuICAgIHRoaXMuX3N0YXRlZnVsID0gYXJncy5zdGF0ZWZ1bCA9PSBudWxsID8gZmFsc2UgOiBhcmdzLnN0YXRlZnVsO1xuICAgIHRoaXMudW5yb2xsID0gYXJncy51bnJvbGwgPT0gbnVsbCA/IGZhbHNlIDogYXJncy51bnJvbGw7XG5cbiAgICB0aGlzLnN1cHBvcnRzTWFza2luZyA9IHRydWU7XG4gICAgdGhpcy5pbnB1dFNwZWMgPSBbbmV3IElucHV0U3BlYyh7bmRpbTogM30pXTtcbiAgICB0aGlzLnN0YXRlU3BlYyA9IG51bGw7XG4gICAgdGhpcy5zdGF0ZXNfID0gbnVsbDtcbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgY29uc3RhbnRzU3BlYyBhbmQgbnVtQ29uc3RhbnRzLlxuICAgIHRoaXMubnVtQ29uc3RhbnRzID0gbnVsbDtcbiAgICAvLyBUT0RPKGNhaXMpOiBMb29rIGludG8gdGhlIHVzZSBvZiBpbml0aWFsX3N0YXRlIGluIHRoZSBrd2FyZ3Mgb2YgdGhlXG4gICAgLy8gICBjb25zdHJ1Y3Rvci5cblxuICAgIHRoaXMua2VwdFN0YXRlcyA9IFtdO1xuICB9XG5cbiAgLy8gUG9ydGluZyBOb3RlOiBUaGlzIGlzIHRoZSBlcXVpdmFsZW50IG9mIGBSTk4uc3RhdGVzYCBwcm9wZXJ0eSBnZXR0ZXIgaW5cbiAgLy8gICBQeUtlcmFzLlxuICBnZXRTdGF0ZXMoKTogVGVuc29yW10ge1xuICAgIGlmICh0aGlzLnN0YXRlc18gPT0gbnVsbCkge1xuICAgICAgY29uc3QgbnVtU3RhdGVzID1cbiAgICAgICAgICBBcnJheS5pc0FycmF5KHRoaXMuY2VsbC5zdGF0ZVNpemUpID8gdGhpcy5jZWxsLnN0YXRlU2l6ZS5sZW5ndGggOiAxO1xuICAgICAgcmV0dXJuIG1hdGhfdXRpbHMucmFuZ2UoMCwgbnVtU3RhdGVzKS5tYXAoeCA9PiBudWxsKTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHRoaXMuc3RhdGVzXztcbiAgICB9XG4gIH1cblxuICAvLyBQb3J0aW5nIE5vdGU6IFRoaXMgaXMgdGhlIGVxdWl2YWxlbnQgb2YgdGhlIGBSTk4uc3RhdGVzYCBwcm9wZXJ0eSBzZXR0ZXIgaW5cbiAgLy8gICBQeUtlcmFzLlxuICBzZXRTdGF0ZXMoc3RhdGVzOiBUZW5zb3JbXSk6IHZvaWQge1xuICAgIHRoaXMuc3RhdGVzXyA9IHN0YXRlcztcbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XG4gICAgaWYgKGlzQXJyYXlPZlNoYXBlcyhpbnB1dFNoYXBlKSkge1xuICAgICAgaW5wdXRTaGFwZSA9IChpbnB1dFNoYXBlIGFzIFNoYXBlW10pWzBdO1xuICAgIH1cbiAgICBpbnB1dFNoYXBlID0gaW5wdXRTaGFwZSBhcyBTaGFwZTtcblxuICAgIC8vIFRPRE8oY2Fpcyk6IFJlbW92ZSB0aGUgY2FzdGluZyBvbmNlIHN0YWNrZWQgUk5OIGNlbGxzIGJlY29tZSBzdXBwb3J0ZWQuXG4gICAgbGV0IHN0YXRlU2l6ZSA9IHRoaXMuY2VsbC5zdGF0ZVNpemU7XG4gICAgaWYgKCFBcnJheS5pc0FycmF5KHN0YXRlU2l6ZSkpIHtcbiAgICAgIHN0YXRlU2l6ZSA9IFtzdGF0ZVNpemVdO1xuICAgIH1cbiAgICBjb25zdCBvdXRwdXREaW0gPSBzdGF0ZVNpemVbMF07XG4gICAgbGV0IG91dHB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdO1xuICAgIGlmICh0aGlzLnJldHVyblNlcXVlbmNlcykge1xuICAgICAgb3V0cHV0U2hhcGUgPSBbaW5wdXRTaGFwZVswXSwgaW5wdXRTaGFwZVsxXSwgb3V0cHV0RGltXTtcbiAgICB9IGVsc2Uge1xuICAgICAgb3V0cHV0U2hhcGUgPSBbaW5wdXRTaGFwZVswXSwgb3V0cHV0RGltXTtcbiAgICB9XG5cbiAgICBpZiAodGhpcy5yZXR1cm5TdGF0ZSkge1xuICAgICAgY29uc3Qgc3RhdGVTaGFwZTogU2hhcGVbXSA9IFtdO1xuICAgICAgZm9yIChjb25zdCBkaW0gb2Ygc3RhdGVTaXplKSB7XG4gICAgICAgIHN0YXRlU2hhcGUucHVzaChbaW5wdXRTaGFwZVswXSwgZGltXSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gW291dHB1dFNoYXBlXS5jb25jYXQoc3RhdGVTaGFwZSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiBvdXRwdXRTaGFwZTtcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlTWFzayhpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwgbWFzaz86IFRlbnNvcnxUZW5zb3JbXSk6IFRlbnNvclxuICAgICAgfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGZjLnRpZHkoKCkgPT4ge1xuICAgICAgaWYgKEFycmF5LmlzQXJyYXkobWFzaykpIHtcbiAgICAgICAgbWFzayA9IG1hc2tbMF07XG4gICAgICB9XG4gICAgICBjb25zdCBvdXRwdXRNYXNrID0gdGhpcy5yZXR1cm5TZXF1ZW5jZXMgPyBtYXNrIDogbnVsbDtcblxuICAgICAgaWYgKHRoaXMucmV0dXJuU3RhdGUpIHtcbiAgICAgICAgY29uc3Qgc3RhdGVNYXNrID0gdGhpcy5zdGF0ZXMubWFwKHMgPT4gbnVsbCk7XG4gICAgICAgIHJldHVybiBbb3V0cHV0TWFza10uY29uY2F0KHN0YXRlTWFzayk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gb3V0cHV0TWFzaztcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBHZXQgdGhlIGN1cnJlbnQgc3RhdGUgdGVuc29ycyBvZiB0aGUgUk5OLlxuICAgKlxuICAgKiBJZiB0aGUgc3RhdGUgaGFzbid0IGJlZW4gc2V0LCByZXR1cm4gYW4gYXJyYXkgb2YgYG51bGxgcyBvZiB0aGUgY29ycmVjdFxuICAgKiBsZW5ndGguXG4gICAqL1xuICBnZXQgc3RhdGVzKCk6IFRlbnNvcltdIHtcbiAgICBpZiAodGhpcy5zdGF0ZXNfID09IG51bGwpIHtcbiAgICAgIGNvbnN0IG51bVN0YXRlcyA9XG4gICAgICAgICAgQXJyYXkuaXNBcnJheSh0aGlzLmNlbGwuc3RhdGVTaXplKSA/IHRoaXMuY2VsbC5zdGF0ZVNpemUubGVuZ3RoIDogMTtcbiAgICAgIGNvbnN0IG91dHB1dDogVGVuc29yW10gPSBbXTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbnVtU3RhdGVzOyArK2kpIHtcbiAgICAgICAgb3V0cHV0LnB1c2gobnVsbCk7XG4gICAgICB9XG4gICAgICByZXR1cm4gb3V0cHV0O1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gdGhpcy5zdGF0ZXNfO1xuICAgIH1cbiAgfVxuXG4gIHNldCBzdGF0ZXMoczogVGVuc29yW10pIHtcbiAgICB0aGlzLnN0YXRlc18gPSBzO1xuICB9XG5cbiAgcHVibGljIG92ZXJyaWRlIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICAvLyBOb3RlIGlucHV0U2hhcGUgd2lsbCBiZSBhbiBBcnJheSBvZiBTaGFwZXMgb2YgaW5pdGlhbCBzdGF0ZXMgYW5kXG4gICAgLy8gY29uc3RhbnRzIGlmIHRoZXNlIGFyZSBwYXNzZWQgaW4gYXBwbHkoKS5cbiAgICBjb25zdCBjb25zdGFudFNoYXBlOiBTaGFwZVtdID0gbnVsbDtcbiAgICBpZiAodGhpcy5udW1Db25zdGFudHMgIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgJ0NvbnN0YW50cyBzdXBwb3J0IGlzIG5vdCBpbXBsZW1lbnRlZCBpbiBSTk4geWV0LicpO1xuICAgIH1cblxuICAgIGlmIChpc0FycmF5T2ZTaGFwZXMoaW5wdXRTaGFwZSkpIHtcbiAgICAgIGlucHV0U2hhcGUgPSAoaW5wdXRTaGFwZSBhcyBTaGFwZVtdKVswXTtcbiAgICB9XG4gICAgaW5wdXRTaGFwZSA9IGlucHV0U2hhcGUgYXMgU2hhcGU7XG5cbiAgICBjb25zdCBiYXRjaFNpemU6IG51bWJlciA9IHRoaXMuc3RhdGVmdWwgPyBpbnB1dFNoYXBlWzBdIDogbnVsbDtcbiAgICBjb25zdCBpbnB1dERpbSA9IGlucHV0U2hhcGUuc2xpY2UoMik7XG4gICAgdGhpcy5pbnB1dFNwZWNbMF0gPSBuZXcgSW5wdXRTcGVjKHtzaGFwZTogW2JhdGNoU2l6ZSwgbnVsbCwgLi4uaW5wdXREaW1dfSk7XG5cbiAgICAvLyBBbGxvdyBjZWxsIChpZiBSTk5DZWxsIExheWVyKSB0byBidWlsZCBiZWZvcmUgd2Ugc2V0IG9yIHZhbGlkYXRlXG4gICAgLy8gc3RhdGVTcGVjLlxuICAgIGNvbnN0IHN0ZXBJbnB1dFNoYXBlID0gW2lucHV0U2hhcGVbMF1dLmNvbmNhdChpbnB1dFNoYXBlLnNsaWNlKDIpKTtcbiAgICBpZiAoY29uc3RhbnRTaGFwZSAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAnQ29uc3RhbnRzIHN1cHBvcnQgaXMgbm90IGltcGxlbWVudGVkIGluIFJOTiB5ZXQuJyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuY2VsbC5idWlsZChzdGVwSW5wdXRTaGFwZSk7XG4gICAgfVxuXG4gICAgLy8gU2V0IG9yIHZhbGlkYXRlIHN0YXRlU3BlYy5cbiAgICBsZXQgc3RhdGVTaXplOiBudW1iZXJbXTtcbiAgICBpZiAoQXJyYXkuaXNBcnJheSh0aGlzLmNlbGwuc3RhdGVTaXplKSkge1xuICAgICAgc3RhdGVTaXplID0gdGhpcy5jZWxsLnN0YXRlU2l6ZTtcbiAgICB9IGVsc2Uge1xuICAgICAgc3RhdGVTaXplID0gW3RoaXMuY2VsbC5zdGF0ZVNpemVdO1xuICAgIH1cblxuICAgIGlmICh0aGlzLnN0YXRlU3BlYyAhPSBudWxsKSB7XG4gICAgICBpZiAoIXV0aWwuYXJyYXlzRXF1YWwoXG4gICAgICAgICAgICAgIHRoaXMuc3RhdGVTcGVjLm1hcChzcGVjID0+IHNwZWMuc2hhcGVbc3BlYy5zaGFwZS5sZW5ndGggLSAxXSksXG4gICAgICAgICAgICAgIHN0YXRlU2l6ZSkpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgQW4gaW5pdGlhbFN0YXRlIHdhcyBwYXNzZWQgdGhhdCBpcyBub3QgY29tcGF0aWJsZSB3aXRoIGAgK1xuICAgICAgICAgICAgYGNlbGwuc3RhdGVTaXplLiBSZWNlaXZlZCBzdGF0ZVNwZWM9JHt0aGlzLnN0YXRlU3BlY307IGAgK1xuICAgICAgICAgICAgYEhvd2V2ZXIgY2VsbC5zdGF0ZVNpemUgaXMgJHt0aGlzLmNlbGwuc3RhdGVTaXplfWApO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLnN0YXRlU3BlYyA9XG4gICAgICAgICAgc3RhdGVTaXplLm1hcChkaW0gPT4gbmV3IElucHV0U3BlYyh7c2hhcGU6IFtudWxsLCBkaW1dfSkpO1xuICAgIH1cbiAgICBpZiAodGhpcy5zdGF0ZWZ1bCkge1xuICAgICAgdGhpcy5yZXNldFN0YXRlcygpO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBSZXNldCB0aGUgc3RhdGUgdGVuc29ycyBvZiB0aGUgUk5OLlxuICAgKlxuICAgKiBJZiB0aGUgYHN0YXRlc2AgYXJndW1lbnQgaXMgYHVuZGVmaW5lZGAgb3IgYG51bGxgLCB3aWxsIHNldCB0aGVcbiAgICogc3RhdGUgdGVuc29yKHMpIG9mIHRoZSBSTk4gdG8gYWxsLXplcm8gdGVuc29ycyBvZiB0aGUgYXBwcm9wcmlhdGVcbiAgICogc2hhcGUocykuXG4gICAqXG4gICAqIElmIGBzdGF0ZXNgIGlzIHByb3ZpZGVkLCB3aWxsIHNldCB0aGUgc3RhdGUgdGVuc29ycyBvZiB0aGUgUk5OIHRvIGl0c1xuICAgKiB2YWx1ZS5cbiAgICpcbiAgICogQHBhcmFtIHN0YXRlcyBPcHRpb25hbCBleHRlcm5hbGx5LXByb3ZpZGVkIGluaXRpYWwgc3RhdGVzLlxuICAgKiBAcGFyYW0gdHJhaW5pbmcgV2hldGhlciB0aGlzIGNhbGwgaXMgZG9uZSBkdXJpbmcgdHJhaW5pbmcuIEZvciBzdGF0ZWZ1bFxuICAgKiAgIFJOTnMsIHRoaXMgYWZmZWN0cyB3aGV0aGVyIHRoZSBvbGQgc3RhdGVzIGFyZSBrZXB0IG9yIGRpc2NhcmRlZC4gSW5cbiAgICogICBwYXJ0aWN1bGFyLCBpZiBgdHJhaW5pbmdgIGlzIGB0cnVlYCwgdGhlIG9sZCBzdGF0ZXMgd2lsbCBiZSBrZXB0IHNvXG4gICAqICAgdGhhdCBzdWJzZXF1ZW50IGJhY2twcm9wZ2F0YWlvbiB0aHJvdWdoIHRpbWUgKEJQVFQpIG1heSB3b3JrIHByb3Blcmx5LlxuICAgKiAgIEVsc2UsIHRoZSBvbGQgc3RhdGVzIHdpbGwgYmUgZGlzY2FyZGVkLlxuICAgKi9cbiAgb3ZlcnJpZGUgcmVzZXRTdGF0ZXMoc3RhdGVzPzogVGVuc29yfFRlbnNvcltdLCB0cmFpbmluZyA9IGZhbHNlKTogdm9pZCB7XG4gICAgdGlkeSgoKSA9PiB7XG4gICAgICBpZiAoIXRoaXMuc3RhdGVmdWwpIHtcbiAgICAgICAgdGhyb3cgbmV3IEF0dHJpYnV0ZUVycm9yKFxuICAgICAgICAgICAgJ0Nhbm5vdCBjYWxsIHJlc2V0U3RhdGVzKCkgb24gYW4gUk5OIExheWVyIHRoYXQgaXMgbm90IHN0YXRlZnVsLicpO1xuICAgICAgfVxuICAgICAgY29uc3QgYmF0Y2hTaXplID0gdGhpcy5pbnB1dFNwZWNbMF0uc2hhcGVbMF07XG4gICAgICBpZiAoYmF0Y2hTaXplID09IG51bGwpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAnSWYgYW4gUk5OIGlzIHN0YXRlZnVsLCBpdCBuZWVkcyB0byBrbm93IGl0cyBiYXRjaCBzaXplLiBTcGVjaWZ5ICcgK1xuICAgICAgICAgICAgJ3RoZSBiYXRjaCBzaXplIG9mIHlvdXIgaW5wdXQgdGVuc29yczogXFxuJyArXG4gICAgICAgICAgICAnLSBJZiB1c2luZyBhIFNlcXVlbnRpYWwgbW9kZWwsIHNwZWNpZnkgdGhlIGJhdGNoIHNpemUgYnkgJyArXG4gICAgICAgICAgICAncGFzc2luZyBhIGBiYXRjaElucHV0U2hhcGVgIG9wdGlvbiB0byB5b3VyIGZpcnN0IGxheWVyLlxcbicgK1xuICAgICAgICAgICAgJy0gSWYgdXNpbmcgdGhlIGZ1bmN0aW9uYWwgQVBJLCBzcGVjaWZ5IHRoZSBiYXRjaCBzaXplIGJ5ICcgK1xuICAgICAgICAgICAgJ3Bhc3NpbmcgYSBgYmF0Y2hTaGFwZWAgb3B0aW9uIHRvIHlvdXIgSW5wdXQgbGF5ZXIuJyk7XG4gICAgICB9XG4gICAgICAvLyBJbml0aWFsaXplIHN0YXRlIGlmIG51bGwuXG4gICAgICBpZiAodGhpcy5zdGF0ZXNfID09IG51bGwpIHtcbiAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5jZWxsLnN0YXRlU2l6ZSkpIHtcbiAgICAgICAgICB0aGlzLnN0YXRlc18gPVxuICAgICAgICAgICAgICB0aGlzLmNlbGwuc3RhdGVTaXplLm1hcChkaW0gPT4gdGZjLnplcm9zKFtiYXRjaFNpemUsIGRpbV0pKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0aGlzLnN0YXRlc18gPSBbdGZjLnplcm9zKFtiYXRjaFNpemUsIHRoaXMuY2VsbC5zdGF0ZVNpemVdKV07XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSBpZiAoc3RhdGVzID09IG51bGwpIHtcbiAgICAgICAgLy8gRGlzcG9zZSBvbGQgc3RhdGUgdGVuc29ycy5cbiAgICAgICAgdGZjLmRpc3Bvc2UodGhpcy5zdGF0ZXNfKTtcbiAgICAgICAgLy8gRm9yIHN0YXRlZnVsIFJOTnMsIGZ1bGx5IGRpc3Bvc2Uga2VwdCBvbGQgc3RhdGVzLlxuICAgICAgICBpZiAodGhpcy5rZXB0U3RhdGVzICE9IG51bGwpIHtcbiAgICAgICAgICB0ZmMuZGlzcG9zZSh0aGlzLmtlcHRTdGF0ZXMpO1xuICAgICAgICAgIHRoaXMua2VwdFN0YXRlcyA9IFtdO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5jZWxsLnN0YXRlU2l6ZSkpIHtcbiAgICAgICAgICB0aGlzLnN0YXRlc18gPVxuICAgICAgICAgICAgICB0aGlzLmNlbGwuc3RhdGVTaXplLm1hcChkaW0gPT4gdGZjLnplcm9zKFtiYXRjaFNpemUsIGRpbV0pKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0aGlzLnN0YXRlc19bMF0gPSB0ZmMuemVyb3MoW2JhdGNoU2l6ZSwgdGhpcy5jZWxsLnN0YXRlU2l6ZV0pO1xuICAgICAgICB9XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBpZiAoIUFycmF5LmlzQXJyYXkoc3RhdGVzKSkge1xuICAgICAgICAgIHN0YXRlcyA9IFtzdGF0ZXNdO1xuICAgICAgICB9XG4gICAgICAgIGlmIChzdGF0ZXMubGVuZ3RoICE9PSB0aGlzLnN0YXRlc18ubGVuZ3RoKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGBMYXllciAke3RoaXMubmFtZX0gZXhwZWN0cyAke3RoaXMuc3RhdGVzXy5sZW5ndGh9IHN0YXRlKHMpLCBgICtcbiAgICAgICAgICAgICAgYGJ1dCBpdCByZWNlaXZlZCAke3N0YXRlcy5sZW5ndGh9IHN0YXRlIHZhbHVlKHMpLiBJbnB1dCBgICtcbiAgICAgICAgICAgICAgYHJlY2VpdmVkOiAke3N0YXRlc31gKTtcbiAgICAgICAgfVxuXG4gICAgICAgIGlmICh0cmFpbmluZyA9PT0gdHJ1ZSkge1xuICAgICAgICAgIC8vIFN0b3JlIG9sZCBzdGF0ZSB0ZW5zb3JzIGZvciBjb21wbGV0ZSBkaXNwb3NhbCBsYXRlciwgaS5lLiwgZHVyaW5nXG4gICAgICAgICAgLy8gdGhlIG5leHQgbm8tYXJnIGNhbGwgdG8gdGhpcyBtZXRob2QuIFdlIGRvIG5vdCBkaXNwb3NlIHRoZSBvbGRcbiAgICAgICAgICAvLyBzdGF0ZXMgaW1tZWRpYXRlbHkgYmVjYXVzZSB0aGF0IEJQVFQgKGFtb25nIG90aGVyIHRoaW5ncykgcmVxdWlyZVxuICAgICAgICAgIC8vIHRoZW0uXG4gICAgICAgICAgdGhpcy5rZXB0U3RhdGVzLnB1c2godGhpcy5zdGF0ZXNfLnNsaWNlKCkpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRmYy5kaXNwb3NlKHRoaXMuc3RhdGVzXyk7XG4gICAgICAgIH1cblxuICAgICAgICBmb3IgKGxldCBpbmRleCA9IDA7IGluZGV4IDwgdGhpcy5zdGF0ZXNfLmxlbmd0aDsgKytpbmRleCkge1xuICAgICAgICAgIGNvbnN0IHZhbHVlID0gc3RhdGVzW2luZGV4XTtcbiAgICAgICAgICBjb25zdCBkaW0gPSBBcnJheS5pc0FycmF5KHRoaXMuY2VsbC5zdGF0ZVNpemUpID9cbiAgICAgICAgICAgICAgdGhpcy5jZWxsLnN0YXRlU2l6ZVtpbmRleF0gOlxuICAgICAgICAgICAgICB0aGlzLmNlbGwuc3RhdGVTaXplO1xuICAgICAgICAgIGNvbnN0IGV4cGVjdGVkU2hhcGUgPSBbYmF0Y2hTaXplLCBkaW1dO1xuICAgICAgICAgIGlmICghdXRpbC5hcnJheXNFcXVhbCh2YWx1ZS5zaGFwZSwgZXhwZWN0ZWRTaGFwZSkpIHtcbiAgICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgICAgIGBTdGF0ZSAke2luZGV4fSBpcyBpbmNvbXBhdGlibGUgd2l0aCBsYXllciAke3RoaXMubmFtZX06IGAgK1xuICAgICAgICAgICAgICAgIGBleHBlY3RlZCBzaGFwZT0ke2V4cGVjdGVkU2hhcGV9LCByZWNlaXZlZCBzaGFwZT0ke1xuICAgICAgICAgICAgICAgICAgICB2YWx1ZS5zaGFwZX1gKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgdGhpcy5zdGF0ZXNfW2luZGV4XSA9IHZhbHVlO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICB0aGlzLnN0YXRlc18gPSB0aGlzLnN0YXRlc18ubWFwKHN0YXRlID0+IHRmYy5rZWVwKHN0YXRlLmNsb25lKCkpKTtcbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFwcGx5KFxuICAgICAgaW5wdXRzOiBUZW5zb3J8VGVuc29yW118U3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXSxcbiAgICAgIGt3YXJncz86IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcnxTeW1ib2xpY1RlbnNvcltdIHtcbiAgICAvLyBUT0RPKGNhaXMpOiBGaWd1cmUgb3V0IHdoZXRoZXIgaW5pdGlhbFN0YXRlIGlzIGluIGt3YXJncyBvciBpbnB1dHMuXG4gICAgbGV0IGluaXRpYWxTdGF0ZTogVGVuc29yW118U3ltYm9saWNUZW5zb3JbXSA9XG4gICAgICAgIGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1snaW5pdGlhbFN0YXRlJ107XG4gICAgbGV0IGNvbnN0YW50czogVGVuc29yW118U3ltYm9saWNUZW5zb3JbXSA9XG4gICAgICAgIGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1snY29uc3RhbnRzJ107XG4gICAgaWYgKGt3YXJncyA9PSBudWxsKSB7XG4gICAgICBrd2FyZ3MgPSB7fTtcbiAgICB9XG5cbiAgICBjb25zdCBzdGFuZGFyZGl6ZWQgPVxuICAgICAgICBzdGFuZGFyZGl6ZUFyZ3MoaW5wdXRzLCBpbml0aWFsU3RhdGUsIGNvbnN0YW50cywgdGhpcy5udW1Db25zdGFudHMpO1xuICAgIGlucHV0cyA9IHN0YW5kYXJkaXplZC5pbnB1dHM7XG4gICAgaW5pdGlhbFN0YXRlID0gc3RhbmRhcmRpemVkLmluaXRpYWxTdGF0ZTtcbiAgICBjb25zdGFudHMgPSBzdGFuZGFyZGl6ZWQuY29uc3RhbnRzO1xuXG4gICAgLy8gSWYgYW55IG9mIGBpbml0aWFsX3N0YXRlYCBvciBgY29uc3RhbnRzYCBhcmUgc3BlY2lmaWVkIGFuZCBhcmVcbiAgICAvLyBgdGYuU3ltYm9saWNUZW5zb3JgcywgdGhlbiBhZGQgdGhlbSB0byB0aGUgaW5wdXRzIGFuZCB0ZW1wb3JhcmlseSBtb2RpZnlcbiAgICAvLyB0aGUgaW5wdXRfc3BlYyB0byBpbmNsdWRlIHRoZW0uXG5cbiAgICBsZXQgYWRkaXRpb25hbElucHV0czogQXJyYXk8VGVuc29yfFN5bWJvbGljVGVuc29yPiA9IFtdO1xuICAgIGxldCBhZGRpdGlvbmFsU3BlY3M6IElucHV0U3BlY1tdID0gW107XG4gICAgaWYgKGluaXRpYWxTdGF0ZSAhPSBudWxsKSB7XG4gICAgICBrd2FyZ3NbJ2luaXRpYWxTdGF0ZSddID0gaW5pdGlhbFN0YXRlO1xuICAgICAgYWRkaXRpb25hbElucHV0cyA9IGFkZGl0aW9uYWxJbnB1dHMuY29uY2F0KGluaXRpYWxTdGF0ZSk7XG4gICAgICB0aGlzLnN0YXRlU3BlYyA9IFtdO1xuICAgICAgZm9yIChjb25zdCBzdGF0ZSBvZiBpbml0aWFsU3RhdGUpIHtcbiAgICAgICAgdGhpcy5zdGF0ZVNwZWMucHVzaChuZXcgSW5wdXRTcGVjKHtzaGFwZTogc3RhdGUuc2hhcGV9KSk7XG4gICAgICB9XG4gICAgICAvLyBUT0RPKGNhaXMpOiBVc2UgdGhlIGZvbGxvd2luZyBpbnN0ZWFkLlxuICAgICAgLy8gdGhpcy5zdGF0ZVNwZWMgPSBpbml0aWFsU3RhdGUubWFwKHN0YXRlID0+IG5ldyBJbnB1dFNwZWMoe3NoYXBlOlxuICAgICAgLy8gc3RhdGUuc2hhcGV9KSk7XG4gICAgICBhZGRpdGlvbmFsU3BlY3MgPSBhZGRpdGlvbmFsU3BlY3MuY29uY2F0KHRoaXMuc3RhdGVTcGVjKTtcbiAgICB9XG4gICAgaWYgKGNvbnN0YW50cyAhPSBudWxsKSB7XG4gICAgICBrd2FyZ3NbJ2NvbnN0YW50cyddID0gY29uc3RhbnRzO1xuICAgICAgYWRkaXRpb25hbElucHV0cyA9IGFkZGl0aW9uYWxJbnB1dHMuY29uY2F0KGNvbnN0YW50cyk7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgdGhpcy5jb25zdGFudHNTcGVjLlxuICAgICAgdGhpcy5udW1Db25zdGFudHMgPSBjb25zdGFudHMubGVuZ3RoO1xuICAgIH1cblxuICAgIGNvbnN0IGlzVGVuc29yID0gYWRkaXRpb25hbElucHV0c1swXSBpbnN0YW5jZW9mIFN5bWJvbGljVGVuc29yO1xuICAgIGlmIChpc1RlbnNvcikge1xuICAgICAgLy8gQ29tcHV0ZSBmdWxsIGlucHV0IHNwZWMsIGluY2x1ZGluZyBzdGF0ZSBhbmQgY29uc3RhbnRzLlxuICAgICAgY29uc3QgZnVsbElucHV0ID1cbiAgICAgICAgICBbaW5wdXRzXS5jb25jYXQoYWRkaXRpb25hbElucHV0cykgYXMgVGVuc29yW10gfCBTeW1ib2xpY1RlbnNvcltdO1xuICAgICAgY29uc3QgZnVsbElucHV0U3BlYyA9IHRoaXMuaW5wdXRTcGVjLmNvbmNhdChhZGRpdGlvbmFsU3BlY3MpO1xuICAgICAgLy8gUGVyZm9ybSB0aGUgY2FsbCB3aXRoIHRlbXBvcmFyaWx5IHJlcGxhY2VkIGlucHV0U3BlYy5cbiAgICAgIGNvbnN0IG9yaWdpbmFsSW5wdXRTcGVjID0gdGhpcy5pbnB1dFNwZWM7XG4gICAgICB0aGlzLmlucHV0U3BlYyA9IGZ1bGxJbnB1dFNwZWM7XG4gICAgICBjb25zdCBvdXRwdXQgPSBzdXBlci5hcHBseShmdWxsSW5wdXQsIGt3YXJncyk7XG4gICAgICB0aGlzLmlucHV0U3BlYyA9IG9yaWdpbmFsSW5wdXRTcGVjO1xuICAgICAgcmV0dXJuIG91dHB1dDtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHN1cGVyLmFwcGx5KGlucHV0cywga3dhcmdzKTtcbiAgICB9XG4gIH1cblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gIG92ZXJyaWRlIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICAvLyBJbnB1dCBzaGFwZTogYFtzYW1wbGVzLCB0aW1lIChwYWRkZWQgd2l0aCB6ZXJvcyksIGlucHV0X2RpbV1gLlxuICAgIC8vIE5vdGUgdGhhdCB0aGUgLmJ1aWxkKCkgbWV0aG9kIG9mIHN1YmNsYXNzZXMgKiptdXN0KiogZGVmaW5lXG4gICAgLy8gdGhpcy5pbnB1dFNwZWMgYW5kIHRoaXMuc3RhdGVTcGVjIG93aXRoIGNvbXBsZXRlIGlucHV0IHNoYXBlcy5cbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBtYXNrID0ga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydtYXNrJ10gYXMgVGVuc29yO1xuICAgICAgY29uc3QgdHJhaW5pbmcgPSBrd2FyZ3MgPT0gbnVsbCA/IG51bGwgOiBrd2FyZ3NbJ3RyYWluaW5nJ107XG4gICAgICBsZXQgaW5pdGlhbFN0YXRlOiBUZW5zb3JbXSA9XG4gICAgICAgICAga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydpbml0aWFsU3RhdGUnXTtcblxuICAgICAgaW5wdXRzID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgICAgaWYgKGluaXRpYWxTdGF0ZSA9PSBudWxsKSB7XG4gICAgICAgIGlmICh0aGlzLnN0YXRlZnVsKSB7XG4gICAgICAgICAgaW5pdGlhbFN0YXRlID0gdGhpcy5zdGF0ZXNfO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGluaXRpYWxTdGF0ZSA9IHRoaXMuZ2V0SW5pdGlhbFN0YXRlKGlucHV0cyk7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgY29uc3QgbnVtU3RhdGVzID1cbiAgICAgICAgICBBcnJheS5pc0FycmF5KHRoaXMuY2VsbC5zdGF0ZVNpemUpID8gdGhpcy5jZWxsLnN0YXRlU2l6ZS5sZW5ndGggOiAxO1xuICAgICAgaWYgKGluaXRpYWxTdGF0ZS5sZW5ndGggIT09IG51bVN0YXRlcykge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBSTk4gTGF5ZXIgaGFzICR7bnVtU3RhdGVzfSBzdGF0ZShzKSBidXQgd2FzIHBhc3NlZCBgICtcbiAgICAgICAgICAgIGAke2luaXRpYWxTdGF0ZS5sZW5ndGh9IGluaXRpYWwgc3RhdGUocykuYCk7XG4gICAgICB9XG4gICAgICBpZiAodGhpcy51bnJvbGwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgJ0lnbm9yaW5nIHVucm9sbCA9IHRydWUgZm9yIFJOTiBsYXllciwgZHVlIHRvIGltcGVyYXRpdmUgYmFja2VuZC4nKTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgY2VsbENhbGxLd2FyZ3M6IEt3YXJncyA9IHt0cmFpbmluZ307XG5cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBzdXBwb3J0IGZvciBjb25zdGFudHMuXG4gICAgICBjb25zdCBzdGVwID0gKGlucHV0czogVGVuc29yLCBzdGF0ZXM6IFRlbnNvcltdKSA9PiB7XG4gICAgICAgIC8vIGBpbnB1dHNgIGFuZCBgc3RhdGVzYCBhcmUgY29uY2F0ZW5hdGVkIHRvIGZvcm0gYSBzaW5nbGUgYEFycmF5YCBvZlxuICAgICAgICAvLyBgdGYuVGVuc29yYHMgYXMgdGhlIGlucHV0IHRvIGBjZWxsLmNhbGwoKWAuXG4gICAgICAgIGNvbnN0IG91dHB1dHMgPVxuICAgICAgICAgICAgdGhpcy5jZWxsLmNhbGwoW2lucHV0c10uY29uY2F0KHN0YXRlcyksIGNlbGxDYWxsS3dhcmdzKSBhcyBUZW5zb3JbXTtcbiAgICAgICAgLy8gTWFyc2hhbGwgdGhlIHJldHVybiB2YWx1ZSBpbnRvIG91dHB1dCBhbmQgbmV3IHN0YXRlcy5cbiAgICAgICAgcmV0dXJuIFtvdXRwdXRzWzBdLCBvdXRwdXRzLnNsaWNlKDEpXSBhcyBbVGVuc29yLCBUZW5zb3JbXV07XG4gICAgICB9O1xuXG4gICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgc3VwcG9ydCBmb3IgY29uc3RhbnRzLlxuXG4gICAgICBjb25zdCBybm5PdXRwdXRzID1cbiAgICAgICAgICBybm4oc3RlcCwgaW5wdXRzLCBpbml0aWFsU3RhdGUsIHRoaXMuZ29CYWNrd2FyZHMsIG1hc2ssIG51bGwsXG4gICAgICAgICAgICAgIHRoaXMudW5yb2xsLCB0aGlzLnJldHVyblNlcXVlbmNlcyk7XG4gICAgICBjb25zdCBsYXN0T3V0cHV0ID0gcm5uT3V0cHV0c1swXTtcbiAgICAgIGNvbnN0IG91dHB1dHMgPSBybm5PdXRwdXRzWzFdO1xuICAgICAgY29uc3Qgc3RhdGVzID0gcm5uT3V0cHV0c1syXTtcblxuICAgICAgaWYgKHRoaXMuc3RhdGVmdWwpIHtcbiAgICAgICAgdGhpcy5yZXNldFN0YXRlcyhzdGF0ZXMsIHRyYWluaW5nKTtcbiAgICAgIH1cblxuICAgICAgY29uc3Qgb3V0cHV0ID0gdGhpcy5yZXR1cm5TZXF1ZW5jZXMgPyBvdXRwdXRzIDogbGFzdE91dHB1dDtcblxuICAgICAgLy8gVE9ETyhjYWlzKTogUG9ycGVydHkgc2V0IGxlYXJuaW5nIHBoYXNlIGZsYWcuXG5cbiAgICAgIGlmICh0aGlzLnJldHVyblN0YXRlKSB7XG4gICAgICAgIHJldHVybiBbb3V0cHV0XS5jb25jYXQoc3RhdGVzKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiBvdXRwdXQ7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICBnZXRJbml0aWFsU3RhdGUoaW5wdXRzOiBUZW5zb3IpOiBUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgLy8gQnVpbGQgYW4gYWxsLXplcm8gdGVuc29yIG9mIHNoYXBlIFtzYW1wbGVzLCBvdXRwdXREaW1dLlxuICAgICAgLy8gW1NhbXBsZXMsIHRpbWVTdGVwcywgaW5wdXREaW1dLlxuICAgICAgbGV0IGluaXRpYWxTdGF0ZSA9IHRmYy56ZXJvcyhpbnB1dHMuc2hhcGUpO1xuICAgICAgLy8gW1NhbXBsZXNdLlxuICAgICAgaW5pdGlhbFN0YXRlID0gdGZjLnN1bShpbml0aWFsU3RhdGUsIFsxLCAyXSk7XG4gICAgICBpbml0aWFsU3RhdGUgPSBLLmV4cGFuZERpbXMoaW5pdGlhbFN0YXRlKTsgIC8vIFtTYW1wbGVzLCAxXS5cblxuICAgICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5jZWxsLnN0YXRlU2l6ZSkpIHtcbiAgICAgICAgcmV0dXJuIHRoaXMuY2VsbC5zdGF0ZVNpemUubWFwKFxuICAgICAgICAgICAgZGltID0+IGRpbSA+IDEgPyBLLnRpbGUoaW5pdGlhbFN0YXRlLCBbMSwgZGltXSkgOiBpbml0aWFsU3RhdGUpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIHRoaXMuY2VsbC5zdGF0ZVNpemUgPiAxID9cbiAgICAgICAgICAgIFtLLnRpbGUoaW5pdGlhbFN0YXRlLCBbMSwgdGhpcy5jZWxsLnN0YXRlU2l6ZV0pXSA6XG4gICAgICAgICAgICBbaW5pdGlhbFN0YXRlXTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldCB0cmFpbmFibGVXZWlnaHRzKCk6IExheWVyVmFyaWFibGVbXSB7XG4gICAgaWYgKCF0aGlzLnRyYWluYWJsZSkge1xuICAgICAgcmV0dXJuIFtdO1xuICAgIH1cbiAgICAvLyBQb3J0aW5nIE5vdGU6IEluIFR5cGVTY3JpcHQsIGB0aGlzYCBpcyBhbHdheXMgYW4gaW5zdGFuY2Ugb2YgYExheWVyYC5cbiAgICByZXR1cm4gdGhpcy5jZWxsLnRyYWluYWJsZVdlaWdodHM7XG4gIH1cblxuICBvdmVycmlkZSBnZXQgbm9uVHJhaW5hYmxlV2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIC8vIFBvcnRpbmcgTm90ZTogSW4gVHlwZVNjcmlwdCwgYHRoaXNgIGlzIGFsd2F5cyBhbiBpbnN0YW5jZSBvZiBgTGF5ZXJgLlxuICAgIGlmICghdGhpcy50cmFpbmFibGUpIHtcbiAgICAgIHJldHVybiB0aGlzLmNlbGwud2VpZ2h0cztcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuY2VsbC5ub25UcmFpbmFibGVXZWlnaHRzO1xuICB9XG5cbiAgb3ZlcnJpZGUgc2V0RmFzdFdlaWdodEluaXREdXJpbmdCdWlsZCh2YWx1ZTogYm9vbGVhbikge1xuICAgIHN1cGVyLnNldEZhc3RXZWlnaHRJbml0RHVyaW5nQnVpbGQodmFsdWUpO1xuICAgIGlmICh0aGlzLmNlbGwgIT0gbnVsbCkge1xuICAgICAgdGhpcy5jZWxsLnNldEZhc3RXZWlnaHRJbml0RHVyaW5nQnVpbGQodmFsdWUpO1xuICAgIH1cbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcblxuICAgIGNvbnN0IGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge1xuICAgICAgcmV0dXJuU2VxdWVuY2VzOiB0aGlzLnJldHVyblNlcXVlbmNlcyxcbiAgICAgIHJldHVyblN0YXRlOiB0aGlzLnJldHVyblN0YXRlLFxuICAgICAgZ29CYWNrd2FyZHM6IHRoaXMuZ29CYWNrd2FyZHMsXG4gICAgICBzdGF0ZWZ1bDogdGhpcy5zdGF0ZWZ1bCxcbiAgICAgIHVucm9sbDogdGhpcy51bnJvbGwsXG4gICAgfTtcblxuICAgIGlmICh0aGlzLm51bUNvbnN0YW50cyAhPSBudWxsKSB7XG4gICAgICBjb25maWdbJ251bUNvbnN0YW50cyddID0gdGhpcy5udW1Db25zdGFudHM7XG4gICAgfVxuXG4gICAgY29uc3QgY2VsbENvbmZpZyA9IHRoaXMuY2VsbC5nZXRDb25maWcoKTtcblxuICAgIGlmICh0aGlzLmdldENsYXNzTmFtZSgpID09PSBSTk4uY2xhc3NOYW1lKSB7XG4gICAgICBjb25maWdbJ2NlbGwnXSA9IHtcbiAgICAgICAgJ2NsYXNzTmFtZSc6IHRoaXMuY2VsbC5nZXRDbGFzc05hbWUoKSxcbiAgICAgICAgJ2NvbmZpZyc6IGNlbGxDb25maWcsXG4gICAgICB9IGFzIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdFZhbHVlO1xuICAgIH1cblxuICAgIC8vIHRoaXMgb3JkZXIgaXMgbmVjZXNzYXJ5LCB0byBwcmV2ZW50IGNlbGwgbmFtZSBmcm9tIHJlcGxhY2luZyBsYXllciBuYW1lXG4gICAgcmV0dXJuIHsuLi5jZWxsQ29uZmlnLCAuLi5iYXNlQ29uZmlnLCAuLi5jb25maWd9O1xuICB9XG5cbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBmcm9tQ29uZmlnPFQgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZT4oXG4gICAgICBjbHM6IHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlQ29uc3RydWN0b3I8VD4sXG4gICAgICBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCxcbiAgICAgIGN1c3RvbU9iamVjdHMgPSB7fSBhcyBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QpOiBUIHtcbiAgICBjb25zdCBjZWxsQ29uZmlnID0gY29uZmlnWydjZWxsJ10gYXMgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0O1xuICAgIGNvbnN0IGNlbGwgPSBkZXNlcmlhbGl6ZShjZWxsQ29uZmlnLCBjdXN0b21PYmplY3RzKSBhcyBSTk5DZWxsO1xuICAgIHJldHVybiBuZXcgY2xzKE9iamVjdC5hc3NpZ24oY29uZmlnLCB7Y2VsbH0pKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFJOTik7XG5cbi8vIFBvcnRpbmcgTm90ZTogVGhpcyBpcyBhIGNvbW1vbiBwYXJlbnQgY2xhc3MgZm9yIFJOTiBjZWxscy4gVGhlcmUgaXMgbm9cbi8vIGVxdWl2YWxlbnQgb2YgdGhpcyBpbiBQeUtlcmFzLiBIYXZpbmcgYSBjb21tb24gcGFyZW50IGNsYXNzIGZvcmdvZXMgdGhlXG4vLyAgbmVlZCBmb3IgYGhhc19hdHRyKGNlbGwsIC4uLilgIGNoZWNrcyBvciBpdHMgVHlwZVNjcmlwdCBlcXVpdmFsZW50LlxuLyoqXG4gKiBBbiBSTk5DZWxsIGxheWVyLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdMYXllcnMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gKi9cbmV4cG9ydCBhYnN0cmFjdCBjbGFzcyBSTk5DZWxsIGV4dGVuZHMgTGF5ZXIge1xuICAvKipcbiAgICogU2l6ZShzKSBvZiB0aGUgc3RhdGVzLlxuICAgKiBGb3IgUk5OIGNlbGxzIHdpdGggb25seSBhIHNpbmdsZSBzdGF0ZSwgdGhpcyBpcyBhIHNpbmdsZSBpbnRlZ2VyLlxuICAgKi9cbiAgLy8gU2VlXG4gIC8vIGh0dHBzOi8vd3d3LnR5cGVzY3JpcHRsYW5nLm9yZy9kb2NzL2hhbmRib29rL3JlbGVhc2Utbm90ZXMvdHlwZXNjcmlwdC00LTAuaHRtbCNwcm9wZXJ0aWVzLW92ZXJyaWRpbmctYWNjZXNzb3JzLWFuZC12aWNlLXZlcnNhLWlzLWFuLWVycm9yXG4gIHB1YmxpYyBhYnN0cmFjdCBzdGF0ZVNpemU6IG51bWJlcnxudW1iZXJbXTtcbiAgcHVibGljIGRyb3BvdXRNYXNrOiBUZW5zb3J8VGVuc29yW107XG4gIHB1YmxpYyByZWN1cnJlbnREcm9wb3V0TWFzazogVGVuc29yfFRlbnNvcltdO1xufVxuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgU2ltcGxlUk5OQ2VsbExheWVyQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIC8qKlxuICAgKiB1bml0czogUG9zaXRpdmUgaW50ZWdlciwgZGltZW5zaW9uYWxpdHkgb2YgdGhlIG91dHB1dCBzcGFjZS5cbiAgICovXG4gIHVuaXRzOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIEFjdGl2YXRpb24gZnVuY3Rpb24gdG8gdXNlLlxuICAgKiBEZWZhdWx0OiBoeXBlcmJvbGljIHRhbmdlbnQgKCd0YW5oJykuXG4gICAqIElmIHlvdSBwYXNzIGBudWxsYCwgICdsaW5lYXInIGFjdGl2YXRpb24gd2lsbCBiZSBhcHBsaWVkLlxuICAgKi9cbiAgYWN0aXZhdGlvbj86IEFjdGl2YXRpb25JZGVudGlmaWVyO1xuXG4gIC8qKlxuICAgKiBXaGV0aGVyIHRoZSBsYXllciB1c2VzIGEgYmlhcyB2ZWN0b3IuXG4gICAqL1xuICB1c2VCaWFzPzogYm9vbGVhbjtcblxuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBga2VybmVsYCB3ZWlnaHRzIG1hdHJpeCwgdXNlZCBmb3IgdGhlIGxpbmVhclxuICAgKiB0cmFuc2Zvcm1hdGlvbiBvZiB0aGUgaW5wdXRzLlxuICAgKi9cbiAga2VybmVsSW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcklkZW50aWZpZXJ8SW5pdGlhbGl6ZXI7XG5cbiAgLyoqXG4gICAqIEluaXRpYWxpemVyIGZvciB0aGUgYHJlY3VycmVudEtlcm5lbGAgd2VpZ2h0cyBtYXRyaXgsIHVzZWQgZm9yXG4gICAqIGxpbmVhciB0cmFuc2Zvcm1hdGlvbiBvZiB0aGUgcmVjdXJyZW50IHN0YXRlLlxuICAgKi9cbiAgcmVjdXJyZW50SW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcklkZW50aWZpZXJ8SW5pdGlhbGl6ZXI7XG5cbiAgLyoqXG4gICAqIEluaXRpYWxpemVyIGZvciB0aGUgYmlhcyB2ZWN0b3IuXG4gICAqL1xuICBiaWFzSW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcklkZW50aWZpZXJ8SW5pdGlhbGl6ZXI7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGBrZXJuZWxgIHdlaWdodHMgbWF0cml4LlxuICAgKi9cbiAga2VybmVsUmVndWxhcml6ZXI/OiBSZWd1bGFyaXplcklkZW50aWZpZXJ8UmVndWxhcml6ZXI7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGByZWN1cnJlbnRfa2VybmVsYCB3ZWlnaHRzIG1hdHJpeC5cbiAgICovXG4gIHJlY3VycmVudFJlZ3VsYXJpemVyPzogUmVndWxhcml6ZXJJZGVudGlmaWVyfFJlZ3VsYXJpemVyO1xuXG4gIC8qKlxuICAgKiBSZWd1bGFyaXplciBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBiaWFzIHZlY3Rvci5cbiAgICovXG4gIGJpYXNSZWd1bGFyaXplcj86IFJlZ3VsYXJpemVySWRlbnRpZmllcnxSZWd1bGFyaXplcjtcblxuICAvKipcbiAgICogQ29uc3RyYWludCBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBga2VybmVsYCB3ZWlnaHRzIG1hdHJpeC5cbiAgICovXG4gIGtlcm5lbENvbnN0cmFpbnQ/OiBDb25zdHJhaW50SWRlbnRpZmllcnxDb25zdHJhaW50O1xuXG4gIC8qKlxuICAgKiBDb25zdHJhaW50IGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGByZWN1cnJlbnRLZXJuZWxgIHdlaWdodHMgbWF0cml4LlxuICAgKi9cbiAgcmVjdXJyZW50Q29uc3RyYWludD86IENvbnN0cmFpbnRJZGVudGlmaWVyfENvbnN0cmFpbnQ7XG5cbiAgLyoqXG4gICAqIENvbnN0cmFpbnQgZnVuY3Rpb24gYXBwbGllZCB0byB0aGUgYmlhcyB2ZWN0b3IuXG4gICAqL1xuICBiaWFzQ29uc3RyYWludD86IENvbnN0cmFpbnRJZGVudGlmaWVyfENvbnN0cmFpbnQ7XG5cbiAgLyoqXG4gICAqIEZsb2F0IG51bWJlciBiZXR3ZWVuIDAgYW5kIDEuIEZyYWN0aW9uIG9mIHRoZSB1bml0cyB0byBkcm9wIGZvciB0aGUgbGluZWFyXG4gICAqIHRyYW5zZm9ybWF0aW9uIG9mIHRoZSBpbnB1dHMuXG4gICAqL1xuICBkcm9wb3V0PzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBGbG9hdCBudW1iZXIgYmV0d2VlbiAwIGFuZCAxLiBGcmFjdGlvbiBvZiB0aGUgdW5pdHMgdG8gZHJvcCBmb3IgdGhlIGxpbmVhclxuICAgKiB0cmFuc2Zvcm1hdGlvbiBvZiB0aGUgcmVjdXJyZW50IHN0YXRlLlxuICAgKi9cbiAgcmVjdXJyZW50RHJvcG91dD86IG51bWJlcjtcblxuICAvKipcbiAgICogVGhpcyBpcyBhZGRlZCBmb3IgdGVzdCBESSBwdXJwb3NlLlxuICAgKi9cbiAgZHJvcG91dEZ1bmM/OiBGdW5jdGlvbjtcbn1cblxuZXhwb3J0IGNsYXNzIFNpbXBsZVJOTkNlbGwgZXh0ZW5kcyBSTk5DZWxsIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnU2ltcGxlUk5OQ2VsbCc7XG4gIHJlYWRvbmx5IHVuaXRzOiBudW1iZXI7XG4gIHJlYWRvbmx5IGFjdGl2YXRpb246IEFjdGl2YXRpb247XG4gIHJlYWRvbmx5IHVzZUJpYXM6IGJvb2xlYW47XG5cbiAgcmVhZG9ubHkga2VybmVsSW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICByZWFkb25seSByZWN1cnJlbnRJbml0aWFsaXplcjogSW5pdGlhbGl6ZXI7XG4gIHJlYWRvbmx5IGJpYXNJbml0aWFsaXplcjogSW5pdGlhbGl6ZXI7XG5cbiAgcmVhZG9ubHkga2VybmVsQ29uc3RyYWludDogQ29uc3RyYWludDtcbiAgcmVhZG9ubHkgcmVjdXJyZW50Q29uc3RyYWludDogQ29uc3RyYWludDtcbiAgcmVhZG9ubHkgYmlhc0NvbnN0cmFpbnQ6IENvbnN0cmFpbnQ7XG5cbiAgcmVhZG9ubHkga2VybmVsUmVndWxhcml6ZXI6IFJlZ3VsYXJpemVyO1xuICByZWFkb25seSByZWN1cnJlbnRSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG4gIHJlYWRvbmx5IGJpYXNSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG5cbiAgcmVhZG9ubHkgZHJvcG91dDogbnVtYmVyO1xuICByZWFkb25seSByZWN1cnJlbnREcm9wb3V0OiBudW1iZXI7XG4gIHJlYWRvbmx5IGRyb3BvdXRGdW5jOiBGdW5jdGlvbjtcblxuICByZWFkb25seSBzdGF0ZVNpemU6IG51bWJlcjtcblxuICBrZXJuZWw6IExheWVyVmFyaWFibGU7XG4gIHJlY3VycmVudEtlcm5lbDogTGF5ZXJWYXJpYWJsZTtcbiAgYmlhczogTGF5ZXJWYXJpYWJsZTtcblxuICByZWFkb25seSBERUZBVUxUX0FDVElWQVRJT04gPSAndGFuaCc7XG4gIHJlYWRvbmx5IERFRkFVTFRfS0VSTkVMX0lOSVRJQUxJWkVSID0gJ2dsb3JvdE5vcm1hbCc7XG4gIHJlYWRvbmx5IERFRkFVTFRfUkVDVVJSRU5UX0lOSVRJQUxJWkVSID0gJ29ydGhvZ29uYWwnO1xuICByZWFkb25seSBERUZBVUxUX0JJQVNfSU5JVElBTElaRVI6IEluaXRpYWxpemVySWRlbnRpZmllciA9ICd6ZXJvcyc7XG5cbiAgY29uc3RydWN0b3IoYXJnczogU2ltcGxlUk5OQ2VsbExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMudW5pdHMgPSBhcmdzLnVuaXRzO1xuICAgIGFzc2VydFBvc2l0aXZlSW50ZWdlcih0aGlzLnVuaXRzLCBgdW5pdHNgKTtcbiAgICB0aGlzLmFjdGl2YXRpb24gPSBnZXRBY3RpdmF0aW9uKFxuICAgICAgICBhcmdzLmFjdGl2YXRpb24gPT0gbnVsbCA/IHRoaXMuREVGQVVMVF9BQ1RJVkFUSU9OIDogYXJncy5hY3RpdmF0aW9uKTtcbiAgICB0aGlzLnVzZUJpYXMgPSBhcmdzLnVzZUJpYXMgPT0gbnVsbCA/IHRydWUgOiBhcmdzLnVzZUJpYXM7XG5cbiAgICB0aGlzLmtlcm5lbEluaXRpYWxpemVyID0gZ2V0SW5pdGlhbGl6ZXIoXG4gICAgICAgIGFyZ3Mua2VybmVsSW5pdGlhbGl6ZXIgfHwgdGhpcy5ERUZBVUxUX0tFUk5FTF9JTklUSUFMSVpFUik7XG4gICAgdGhpcy5yZWN1cnJlbnRJbml0aWFsaXplciA9IGdldEluaXRpYWxpemVyKFxuICAgICAgICBhcmdzLnJlY3VycmVudEluaXRpYWxpemVyIHx8IHRoaXMuREVGQVVMVF9SRUNVUlJFTlRfSU5JVElBTElaRVIpO1xuXG4gICAgdGhpcy5iaWFzSW5pdGlhbGl6ZXIgPVxuICAgICAgICBnZXRJbml0aWFsaXplcihhcmdzLmJpYXNJbml0aWFsaXplciB8fCB0aGlzLkRFRkFVTFRfQklBU19JTklUSUFMSVpFUik7XG5cbiAgICB0aGlzLmtlcm5lbFJlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoYXJncy5rZXJuZWxSZWd1bGFyaXplcik7XG4gICAgdGhpcy5yZWN1cnJlbnRSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3MucmVjdXJyZW50UmVndWxhcml6ZXIpO1xuICAgIHRoaXMuYmlhc1JlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoYXJncy5iaWFzUmVndWxhcml6ZXIpO1xuXG4gICAgdGhpcy5rZXJuZWxDb25zdHJhaW50ID0gZ2V0Q29uc3RyYWludChhcmdzLmtlcm5lbENvbnN0cmFpbnQpO1xuICAgIHRoaXMucmVjdXJyZW50Q29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5yZWN1cnJlbnRDb25zdHJhaW50KTtcbiAgICB0aGlzLmJpYXNDb25zdHJhaW50ID0gZ2V0Q29uc3RyYWludChhcmdzLmJpYXNDb25zdHJhaW50KTtcblxuICAgIHRoaXMuZHJvcG91dCA9IG1hdGhfdXRpbHMubWluKFxuICAgICAgICBbMSwgbWF0aF91dGlscy5tYXgoWzAsIGFyZ3MuZHJvcG91dCA9PSBudWxsID8gMCA6IGFyZ3MuZHJvcG91dF0pXSk7XG4gICAgdGhpcy5yZWN1cnJlbnREcm9wb3V0ID0gbWF0aF91dGlscy5taW4oW1xuICAgICAgMSxcbiAgICAgIG1hdGhfdXRpbHMubWF4KFxuICAgICAgICAgIFswLCBhcmdzLnJlY3VycmVudERyb3BvdXQgPT0gbnVsbCA/IDAgOiBhcmdzLnJlY3VycmVudERyb3BvdXRdKVxuICAgIF0pO1xuICAgIHRoaXMuZHJvcG91dEZ1bmMgPSBhcmdzLmRyb3BvdXRGdW5jO1xuICAgIHRoaXMuc3RhdGVTaXplID0gdGhpcy51bml0cztcbiAgICB0aGlzLmRyb3BvdXRNYXNrID0gbnVsbDtcbiAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrID0gbnVsbDtcbiAgfVxuXG4gIG92ZXJyaWRlIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICBpbnB1dFNoYXBlID0gZ2V0RXhhY3RseU9uZVNoYXBlKGlucHV0U2hhcGUpO1xuICAgIC8vIFRPRE8oY2Fpcyk6IFVzZSByZWd1bGFyaXplci5cbiAgICB0aGlzLmtlcm5lbCA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAna2VybmVsJywgW2lucHV0U2hhcGVbaW5wdXRTaGFwZS5sZW5ndGggLSAxXSwgdGhpcy51bml0c10sIG51bGwsXG4gICAgICAgIHRoaXMua2VybmVsSW5pdGlhbGl6ZXIsIHRoaXMua2VybmVsUmVndWxhcml6ZXIsIHRydWUsXG4gICAgICAgIHRoaXMua2VybmVsQ29uc3RyYWludCk7XG4gICAgdGhpcy5yZWN1cnJlbnRLZXJuZWwgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgJ3JlY3VycmVudF9rZXJuZWwnLCBbdGhpcy51bml0cywgdGhpcy51bml0c10sIG51bGwsXG4gICAgICAgIHRoaXMucmVjdXJyZW50SW5pdGlhbGl6ZXIsIHRoaXMucmVjdXJyZW50UmVndWxhcml6ZXIsIHRydWUsXG4gICAgICAgIHRoaXMucmVjdXJyZW50Q29uc3RyYWludCk7XG4gICAgaWYgKHRoaXMudXNlQmlhcykge1xuICAgICAgdGhpcy5iaWFzID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICAgJ2JpYXMnLCBbdGhpcy51bml0c10sIG51bGwsIHRoaXMuYmlhc0luaXRpYWxpemVyLFxuICAgICAgICAgIHRoaXMuYmlhc1JlZ3VsYXJpemVyLCB0cnVlLCB0aGlzLmJpYXNDb25zdHJhaW50KTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5iaWFzID0gbnVsbDtcbiAgICB9XG4gICAgdGhpcy5idWlsdCA9IHRydWU7XG4gIH1cblxuICAvLyBQb3J0aW5nIE5vdGU6IFB5S2VyYXMnIGVxdWl2YWxlbnQgb2YgdGhpcyBtZXRob2QgdGFrZXMgdHdvIHRlbnNvciBpbnB1dHM6XG4gIC8vICAgYGlucHV0c2AgYW5kIGBzdGF0ZXNgLiBIZXJlLCB0aGUgdHdvIHRlbnNvcnMgYXJlIGNvbWJpbmVkIGludG8gYW5cbiAgLy8gICBgVGVuc29yW11gIEFycmF5IGFzIHRoZSBmaXJzdCBpbnB1dCBhcmd1bWVudC5cbiAgLy8gICBTaW1pbGFybHksIFB5S2VyYXMnIGVxdWl2YWxlbnQgb2YgdGhpcyBtZXRob2QgcmV0dXJucyB0d28gdmFsdWVzOlxuICAvLyAgICBgb3V0cHV0YCBhbmQgYFtvdXRwdXRdYC4gSGVyZSB0aGUgdHdvIGFyZSBjb21iaW5lZCBpbnRvIG9uZSBsZW5ndGgtMlxuICAvLyAgICBgVGVuc29yW11gLCBjb25zaXN0aW5nIG9mIGBvdXRwdXRgIHJlcGVhdGVkLlxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgaW5wdXRzID0gaW5wdXRzIGFzIFRlbnNvcltdO1xuICAgICAgaWYgKGlucHV0cy5sZW5ndGggIT09IDIpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgU2ltcGxlUk5OQ2VsbCBleHBlY3RzIDIgaW5wdXQgVGVuc29ycywgZ290ICR7aW5wdXRzLmxlbmd0aH0uYCk7XG4gICAgICB9XG4gICAgICBsZXQgcHJldk91dHB1dCA9IGlucHV0c1sxXTtcbiAgICAgIGlucHV0cyA9IGlucHV0c1swXTtcbiAgICAgIGNvbnN0IHRyYWluaW5nID0ga3dhcmdzWyd0cmFpbmluZyddID09IG51bGwgPyBmYWxzZSA6IGt3YXJnc1sndHJhaW5pbmcnXTtcblxuICAgICAgaWYgKDAgPCB0aGlzLmRyb3BvdXQgJiYgdGhpcy5kcm9wb3V0IDwgMSAmJiB0aGlzLmRyb3BvdXRNYXNrID09IG51bGwpIHtcbiAgICAgICAgdGhpcy5kcm9wb3V0TWFzayA9IGdlbmVyYXRlRHJvcG91dE1hc2soe1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICBvbmVzOiAoKSA9PiB0ZmMub25lc0xpa2UoaW5wdXRzIGFzIFRlbnNvciksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJhdGU6IHRoaXMuZHJvcG91dCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdHJhaW5pbmcsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRyb3BvdXRGdW5jOiB0aGlzLmRyb3BvdXRGdW5jLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgfSkgYXMgVGVuc29yO1xuICAgICAgfVxuICAgICAgaWYgKDAgPCB0aGlzLnJlY3VycmVudERyb3BvdXQgJiYgdGhpcy5yZWN1cnJlbnREcm9wb3V0IDwgMSAmJlxuICAgICAgICAgIHRoaXMucmVjdXJyZW50RHJvcG91dE1hc2sgPT0gbnVsbCkge1xuICAgICAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrID0gZ2VuZXJhdGVEcm9wb3V0TWFzayh7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9uZXM6ICgpID0+IHRmYy5vbmVzTGlrZShwcmV2T3V0cHV0KSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmF0ZTogdGhpcy5yZWN1cnJlbnREcm9wb3V0LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0cmFpbmluZyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZHJvcG91dEZ1bmM6IHRoaXMuZHJvcG91dEZ1bmMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KSBhcyBUZW5zb3I7XG4gICAgICB9XG4gICAgICBsZXQgaDogVGVuc29yO1xuICAgICAgY29uc3QgZHBNYXNrOiBUZW5zb3IgPSB0aGlzLmRyb3BvdXRNYXNrIGFzIFRlbnNvcjtcbiAgICAgIGNvbnN0IHJlY0RwTWFzazogVGVuc29yID0gdGhpcy5yZWN1cnJlbnREcm9wb3V0TWFzayBhcyBUZW5zb3I7XG4gICAgICBpZiAoZHBNYXNrICE9IG51bGwpIHtcbiAgICAgICAgaCA9IEsuZG90KHRmYy5tdWwoaW5wdXRzLCBkcE1hc2spLCB0aGlzLmtlcm5lbC5yZWFkKCkpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgaCA9IEsuZG90KGlucHV0cywgdGhpcy5rZXJuZWwucmVhZCgpKTtcbiAgICAgIH1cbiAgICAgIGlmICh0aGlzLmJpYXMgIT0gbnVsbCkge1xuICAgICAgICBoID0gSy5iaWFzQWRkKGgsIHRoaXMuYmlhcy5yZWFkKCkpO1xuICAgICAgfVxuICAgICAgaWYgKHJlY0RwTWFzayAhPSBudWxsKSB7XG4gICAgICAgIHByZXZPdXRwdXQgPSB0ZmMubXVsKHByZXZPdXRwdXQsIHJlY0RwTWFzayk7XG4gICAgICB9XG4gICAgICBsZXQgb3V0cHV0ID0gdGZjLmFkZChoLCBLLmRvdChwcmV2T3V0cHV0LCB0aGlzLnJlY3VycmVudEtlcm5lbC5yZWFkKCkpKTtcbiAgICAgIGlmICh0aGlzLmFjdGl2YXRpb24gIT0gbnVsbCkge1xuICAgICAgICBvdXRwdXQgPSB0aGlzLmFjdGl2YXRpb24uYXBwbHkob3V0cHV0KTtcbiAgICAgIH1cblxuICAgICAgLy8gVE9ETyhjYWlzKTogUHJvcGVybHkgc2V0IGxlYXJuaW5nIHBoYXNlIG9uIG91dHB1dCB0ZW5zb3I/XG4gICAgICByZXR1cm4gW291dHB1dCwgb3V0cHV0XTtcbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcblxuICAgIGNvbnN0IGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge1xuICAgICAgdW5pdHM6IHRoaXMudW5pdHMsXG4gICAgICBhY3RpdmF0aW9uOiBzZXJpYWxpemVBY3RpdmF0aW9uKHRoaXMuYWN0aXZhdGlvbiksXG4gICAgICB1c2VCaWFzOiB0aGlzLnVzZUJpYXMsXG4gICAgICBrZXJuZWxJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5rZXJuZWxJbml0aWFsaXplciksXG4gICAgICByZWN1cnJlbnRJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5yZWN1cnJlbnRJbml0aWFsaXplciksXG4gICAgICBiaWFzSW5pdGlhbGl6ZXI6IHNlcmlhbGl6ZUluaXRpYWxpemVyKHRoaXMuYmlhc0luaXRpYWxpemVyKSxcbiAgICAgIGtlcm5lbFJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLmtlcm5lbFJlZ3VsYXJpemVyKSxcbiAgICAgIHJlY3VycmVudFJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLnJlY3VycmVudFJlZ3VsYXJpemVyKSxcbiAgICAgIGJpYXNSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5iaWFzUmVndWxhcml6ZXIpLFxuICAgICAgYWN0aXZpdHlSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5hY3Rpdml0eVJlZ3VsYXJpemVyKSxcbiAgICAgIGtlcm5lbENvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5rZXJuZWxDb25zdHJhaW50KSxcbiAgICAgIHJlY3VycmVudENvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5yZWN1cnJlbnRDb25zdHJhaW50KSxcbiAgICAgIGJpYXNDb25zdHJhaW50OiBzZXJpYWxpemVDb25zdHJhaW50KHRoaXMuYmlhc0NvbnN0cmFpbnQpLFxuICAgICAgZHJvcG91dDogdGhpcy5kcm9wb3V0LFxuICAgICAgcmVjdXJyZW50RHJvcG91dDogdGhpcy5yZWN1cnJlbnREcm9wb3V0LFxuICAgIH07XG5cbiAgICByZXR1cm4gey4uLmJhc2VDb25maWcsIC4uLmNvbmZpZ307XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhTaW1wbGVSTk5DZWxsKTtcblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIFNpbXBsZVJOTkxheWVyQXJncyBleHRlbmRzIEJhc2VSTk5MYXllckFyZ3Mge1xuICAvKipcbiAgICogUG9zaXRpdmUgaW50ZWdlciwgZGltZW5zaW9uYWxpdHkgb2YgdGhlIG91dHB1dCBzcGFjZS5cbiAgICovXG4gIHVuaXRzOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIEFjdGl2YXRpb24gZnVuY3Rpb24gdG8gdXNlLlxuICAgKlxuICAgKiBEZWZhdWx0cyB0byAgaHlwZXJib2xpYyB0YW5nZW50IChgdGFuaGApXG4gICAqXG4gICAqIElmIHlvdSBwYXNzIGBudWxsYCwgbm8gYWN0aXZhdGlvbiB3aWxsIGJlIGFwcGxpZWQuXG4gICAqL1xuICBhY3RpdmF0aW9uPzogQWN0aXZhdGlvbklkZW50aWZpZXI7XG5cbiAgLyoqXG4gICAqIFdoZXRoZXIgdGhlIGxheWVyIHVzZXMgYSBiaWFzIHZlY3Rvci5cbiAgICovXG4gIHVzZUJpYXM/OiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBJbml0aWFsaXplciBmb3IgdGhlIGBrZXJuZWxgIHdlaWdodHMgbWF0cml4LCB1c2VkIGZvciB0aGUgbGluZWFyXG4gICAqIHRyYW5zZm9ybWF0aW9uIG9mIHRoZSBpbnB1dHMuXG4gICAqL1xuICBrZXJuZWxJbml0aWFsaXplcj86IEluaXRpYWxpemVySWRlbnRpZmllcnxJbml0aWFsaXplcjtcblxuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBgcmVjdXJyZW50S2VybmVsYCB3ZWlnaHRzIG1hdHJpeCwgdXNlZCBmb3JcbiAgICogbGluZWFyIHRyYW5zZm9ybWF0aW9uIG9mIHRoZSByZWN1cnJlbnQgc3RhdGUuXG4gICAqL1xuICByZWN1cnJlbnRJbml0aWFsaXplcj86IEluaXRpYWxpemVySWRlbnRpZmllcnxJbml0aWFsaXplcjtcblxuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBiaWFzIHZlY3Rvci5cbiAgICovXG4gIGJpYXNJbml0aWFsaXplcj86IEluaXRpYWxpemVySWRlbnRpZmllcnxJbml0aWFsaXplcjtcblxuICAvKipcbiAgICogUmVndWxhcml6ZXIgZnVuY3Rpb24gYXBwbGllZCB0byB0aGUga2VybmVsIHdlaWdodHMgbWF0cml4LlxuICAgKi9cbiAga2VybmVsUmVndWxhcml6ZXI/OiBSZWd1bGFyaXplcklkZW50aWZpZXJ8UmVndWxhcml6ZXI7XG5cbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIHJlY3VycmVudEtlcm5lbCB3ZWlnaHRzIG1hdHJpeC5cbiAgICovXG4gIHJlY3VycmVudFJlZ3VsYXJpemVyPzogUmVndWxhcml6ZXJJZGVudGlmaWVyfFJlZ3VsYXJpemVyO1xuXG4gIC8qKlxuICAgKiBSZWd1bGFyaXplciBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBiaWFzIHZlY3Rvci5cbiAgICovXG4gIGJpYXNSZWd1bGFyaXplcj86IFJlZ3VsYXJpemVySWRlbnRpZmllcnxSZWd1bGFyaXplcjtcblxuICAvKipcbiAgICogQ29uc3RyYWludCBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBrZXJuZWwgd2VpZ2h0cyBtYXRyaXguXG4gICAqL1xuICBrZXJuZWxDb25zdHJhaW50PzogQ29uc3RyYWludElkZW50aWZpZXJ8Q29uc3RyYWludDtcblxuICAvKipcbiAgICogQ29uc3RyYWludCBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSByZWN1cnJlbnRLZXJuZWwgd2VpZ2h0cyBtYXRyaXguXG4gICAqL1xuICByZWN1cnJlbnRDb25zdHJhaW50PzogQ29uc3RyYWludElkZW50aWZpZXJ8Q29uc3RyYWludDtcblxuICAvKipcbiAgICogQ29uc3RyYWludCBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBiaWFzIHZlY3Rvci5cbiAgICovXG4gIGJpYXNDb25zdHJhaW50PzogQ29uc3RyYWludElkZW50aWZpZXJ8Q29uc3RyYWludDtcblxuICAvKipcbiAgICogTnVtYmVyIGJldHdlZW4gMCBhbmQgMS4gRnJhY3Rpb24gb2YgdGhlIHVuaXRzIHRvIGRyb3AgZm9yIHRoZSBsaW5lYXJcbiAgICogdHJhbnNmb3JtYXRpb24gb2YgdGhlIGlucHV0cy5cbiAgICovXG4gIGRyb3BvdXQ/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE51bWJlciBiZXR3ZWVuIDAgYW5kIDEuIEZyYWN0aW9uIG9mIHRoZSB1bml0cyB0byBkcm9wIGZvciB0aGUgbGluZWFyXG4gICAqIHRyYW5zZm9ybWF0aW9uIG9mIHRoZSByZWN1cnJlbnQgc3RhdGUuXG4gICAqL1xuICByZWN1cnJlbnREcm9wb3V0PzogbnVtYmVyO1xuXG4gIC8qKlxuICAgKiBUaGlzIGlzIGFkZGVkIGZvciB0ZXN0IERJIHB1cnBvc2UuXG4gICAqL1xuICBkcm9wb3V0RnVuYz86IEZ1bmN0aW9uO1xufVxuXG4vKipcbiAqIFJOTkxheWVyQ29uZmlnIGlzIGlkZW50aWNhbCB0byBCYXNlUk5OTGF5ZXJDb25maWcsIGV4Y2VwdCBpdCBtYWtlcyB0aGVcbiAqIGBjZWxsYCBwcm9wZXJ0eSByZXF1aXJlZC4gVGhpcyBpbnRlcmZhY2UgaXMgdG8gYmUgdXNlZCB3aXRoIGNvbnN0cnVjdG9yc1xuICogb2YgY29uY3JldGUgUk5OIGxheWVyIHN1YnR5cGVzLlxuICovXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgUk5OTGF5ZXJBcmdzIGV4dGVuZHMgQmFzZVJOTkxheWVyQXJncyB7XG4gIGNlbGw6IFJOTkNlbGx8Uk5OQ2VsbFtdO1xufVxuXG5leHBvcnQgY2xhc3MgU2ltcGxlUk5OIGV4dGVuZHMgUk5OIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBjbGFzc05hbWUgPSAnU2ltcGxlUk5OJztcbiAgY29uc3RydWN0b3IoYXJnczogU2ltcGxlUk5OTGF5ZXJBcmdzKSB7XG4gICAgYXJncy5jZWxsID0gbmV3IFNpbXBsZVJOTkNlbGwoYXJncyk7XG4gICAgc3VwZXIoYXJncyBhcyBSTk5MYXllckFyZ3MpO1xuICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBhY3Rpdml0eVJlZ3VsYXJpemVyLlxuICB9XG5cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGlmICh0aGlzLmNlbGwuZHJvcG91dE1hc2sgIT0gbnVsbCkge1xuICAgICAgICB0ZmMuZGlzcG9zZSh0aGlzLmNlbGwuZHJvcG91dE1hc2spO1xuICAgICAgICB0aGlzLmNlbGwuZHJvcG91dE1hc2sgPSBudWxsO1xuICAgICAgfVxuICAgICAgaWYgKHRoaXMuY2VsbC5yZWN1cnJlbnREcm9wb3V0TWFzayAhPSBudWxsKSB7XG4gICAgICAgIHRmYy5kaXNwb3NlKHRoaXMuY2VsbC5yZWN1cnJlbnREcm9wb3V0TWFzayk7XG4gICAgICAgIHRoaXMuY2VsbC5yZWN1cnJlbnREcm9wb3V0TWFzayA9IG51bGw7XG4gICAgICB9XG4gICAgICBjb25zdCBtYXNrID0ga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydtYXNrJ107XG4gICAgICBjb25zdCB0cmFpbmluZyA9IGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1sndHJhaW5pbmcnXTtcbiAgICAgIGNvbnN0IGluaXRpYWxTdGF0ZTogVGVuc29yW10gPVxuICAgICAgICAgIGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1snaW5pdGlhbFN0YXRlJ107XG4gICAgICByZXR1cm4gc3VwZXIuY2FsbChpbnB1dHMsIHttYXNrLCB0cmFpbmluZywgaW5pdGlhbFN0YXRlfSk7XG4gICAgfSk7XG4gIH1cblxuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIG92ZXJyaWRlIGZyb21Db25maWc8VCBleHRlbmRzIHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlPihcbiAgICAgIGNsczogc2VyaWFsaXphdGlvbi5TZXJpYWxpemFibGVDb25zdHJ1Y3RvcjxUPixcbiAgICAgIGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0KTogVCB7XG4gICAgcmV0dXJuIG5ldyBjbHMoY29uZmlnKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFNpbXBsZVJOTik7XG5cbi8vIFBvcnRpbmcgTm90ZTogU2luY2UgdGhpcyBpcyBhIHN1cGVyc2V0IG9mIFNpbXBsZVJOTkxheWVyQ29uZmlnLCB3ZSBleHRlbmRcbi8vICAgdGhhdCBpbnRlcmZhY2UgaW5zdGVhZCBvZiByZXBlYXRpbmcgdGhlIGZpZWxkcy5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBHUlVDZWxsTGF5ZXJBcmdzIGV4dGVuZHMgU2ltcGxlUk5OQ2VsbExheWVyQXJncyB7XG4gIC8qKlxuICAgKiBBY3RpdmF0aW9uIGZ1bmN0aW9uIHRvIHVzZSBmb3IgdGhlIHJlY3VycmVudCBzdGVwLlxuICAgKlxuICAgKiBEZWZhdWx0cyB0byBoYXJkIHNpZ21vaWQgKGBoYXJkU2lnbW9pZGApLlxuICAgKlxuICAgKiBJZiBgbnVsbGAsIG5vIGFjdGl2YXRpb24gaXMgYXBwbGllZC5cbiAgICovXG4gIHJlY3VycmVudEFjdGl2YXRpb24/OiBBY3RpdmF0aW9uSWRlbnRpZmllcjtcblxuICAvKipcbiAgICogSW1wbGVtZW50YXRpb24gbW9kZSwgZWl0aGVyIDEgb3IgMi5cbiAgICpcbiAgICogTW9kZSAxIHdpbGwgc3RydWN0dXJlIGl0cyBvcGVyYXRpb25zIGFzIGEgbGFyZ2VyIG51bWJlciBvZlxuICAgKiAgIHNtYWxsZXIgZG90IHByb2R1Y3RzIGFuZCBhZGRpdGlvbnMuXG4gICAqXG4gICAqIE1vZGUgMiB3aWxsIGJhdGNoIHRoZW0gaW50byBmZXdlciwgbGFyZ2VyIG9wZXJhdGlvbnMuIFRoZXNlIG1vZGVzIHdpbGxcbiAgICogaGF2ZSBkaWZmZXJlbnQgcGVyZm9ybWFuY2UgcHJvZmlsZXMgb24gZGlmZmVyZW50IGhhcmR3YXJlIGFuZFxuICAgKiBmb3IgZGlmZmVyZW50IGFwcGxpY2F0aW9ucy5cbiAgICpcbiAgICogTm90ZTogRm9yIHN1cGVyaW9yIHBlcmZvcm1hbmNlLCBUZW5zb3JGbG93LmpzIGFsd2F5cyB1c2VzIGltcGxlbWVudGF0aW9uXG4gICAqIDIsIHJlZ2FyZGxlc3Mgb2YgdGhlIGFjdHVhbCB2YWx1ZSBvZiB0aGlzIGNvbmZpZ3VyYXRpb24gZmllbGQuXG4gICAqL1xuICBpbXBsZW1lbnRhdGlvbj86IG51bWJlcjtcblxuICAvKipcbiAgICogR1JVIGNvbnZlbnRpb24gKHdoZXRoZXIgdG8gYXBwbHkgcmVzZXQgZ2F0ZSBhZnRlciBvciBiZWZvcmUgbWF0cml4XG4gICAqIG11bHRpcGxpY2F0aW9uKS4gZmFsc2UgPSBcImJlZm9yZVwiLCB0cnVlID0gXCJhZnRlclwiIChvbmx5IGZhbHNlIGlzXG4gICAqIHN1cHBvcnRlZCkuXG4gICAqL1xuICByZXNldEFmdGVyPzogYm9vbGVhbjtcbn1cblxuZXhwb3J0IGNsYXNzIEdSVUNlbGwgZXh0ZW5kcyBSTk5DZWxsIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnR1JVQ2VsbCc7XG4gIHJlYWRvbmx5IHVuaXRzOiBudW1iZXI7XG4gIHJlYWRvbmx5IGFjdGl2YXRpb246IEFjdGl2YXRpb247XG4gIHJlYWRvbmx5IHJlY3VycmVudEFjdGl2YXRpb246IEFjdGl2YXRpb247XG4gIHJlYWRvbmx5IHVzZUJpYXM6IGJvb2xlYW47XG5cbiAgcmVhZG9ubHkga2VybmVsSW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICByZWFkb25seSByZWN1cnJlbnRJbml0aWFsaXplcjogSW5pdGlhbGl6ZXI7XG4gIHJlYWRvbmx5IGJpYXNJbml0aWFsaXplcjogSW5pdGlhbGl6ZXI7XG5cbiAgcmVhZG9ubHkga2VybmVsUmVndWxhcml6ZXI6IFJlZ3VsYXJpemVyO1xuICByZWFkb25seSByZWN1cnJlbnRSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG4gIHJlYWRvbmx5IGJpYXNSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG5cbiAgcmVhZG9ubHkga2VybmVsQ29uc3RyYWludDogQ29uc3RyYWludDtcbiAgcmVhZG9ubHkgcmVjdXJyZW50Q29uc3RyYWludDogQ29uc3RyYWludDtcbiAgcmVhZG9ubHkgYmlhc0NvbnN0cmFpbnQ6IENvbnN0cmFpbnQ7XG5cbiAgcmVhZG9ubHkgZHJvcG91dDogbnVtYmVyO1xuICByZWFkb25seSByZWN1cnJlbnREcm9wb3V0OiBudW1iZXI7XG4gIHJlYWRvbmx5IGRyb3BvdXRGdW5jOiBGdW5jdGlvbjtcblxuICByZWFkb25seSBzdGF0ZVNpemU6IG51bWJlcjtcbiAgcmVhZG9ubHkgaW1wbGVtZW50YXRpb246IG51bWJlcjtcblxuICByZWFkb25seSBERUZBVUxUX0FDVElWQVRJT04gPSAndGFuaCc7XG4gIHJlYWRvbmx5IERFRkFVTFRfUkVDVVJSRU5UX0FDVElWQVRJT046IEFjdGl2YXRpb25JZGVudGlmaWVyID0gJ2hhcmRTaWdtb2lkJztcblxuICByZWFkb25seSBERUZBVUxUX0tFUk5FTF9JTklUSUFMSVpFUiA9ICdnbG9yb3ROb3JtYWwnO1xuICByZWFkb25seSBERUZBVUxUX1JFQ1VSUkVOVF9JTklUSUFMSVpFUiA9ICdvcnRob2dvbmFsJztcbiAgcmVhZG9ubHkgREVGQVVMVF9CSUFTX0lOSVRJQUxJWkVSOiBJbml0aWFsaXplcklkZW50aWZpZXIgPSAnemVyb3MnO1xuXG4gIGtlcm5lbDogTGF5ZXJWYXJpYWJsZTtcbiAgcmVjdXJyZW50S2VybmVsOiBMYXllclZhcmlhYmxlO1xuICBiaWFzOiBMYXllclZhcmlhYmxlO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IEdSVUNlbGxMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzKTtcbiAgICBpZiAoYXJncy5yZXNldEFmdGVyKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgR1JVQ2VsbCBkb2VzIG5vdCBzdXBwb3J0IHJlc2V0X2FmdGVyIHBhcmFtZXRlciBzZXQgdG8gdHJ1ZS5gKTtcbiAgICB9XG4gICAgdGhpcy51bml0cyA9IGFyZ3MudW5pdHM7XG4gICAgYXNzZXJ0UG9zaXRpdmVJbnRlZ2VyKHRoaXMudW5pdHMsICd1bml0cycpO1xuICAgIHRoaXMuYWN0aXZhdGlvbiA9IGdldEFjdGl2YXRpb24oXG4gICAgICAgIGFyZ3MuYWN0aXZhdGlvbiA9PT0gdW5kZWZpbmVkID8gdGhpcy5ERUZBVUxUX0FDVElWQVRJT04gOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFyZ3MuYWN0aXZhdGlvbik7XG4gICAgdGhpcy5yZWN1cnJlbnRBY3RpdmF0aW9uID0gZ2V0QWN0aXZhdGlvbihcbiAgICAgICAgYXJncy5yZWN1cnJlbnRBY3RpdmF0aW9uID09PSB1bmRlZmluZWQgP1xuICAgICAgICAgICAgdGhpcy5ERUZBVUxUX1JFQ1VSUkVOVF9BQ1RJVkFUSU9OIDpcbiAgICAgICAgICAgIGFyZ3MucmVjdXJyZW50QWN0aXZhdGlvbik7XG4gICAgdGhpcy51c2VCaWFzID0gYXJncy51c2VCaWFzID09IG51bGwgPyB0cnVlIDogYXJncy51c2VCaWFzO1xuXG4gICAgdGhpcy5rZXJuZWxJbml0aWFsaXplciA9IGdldEluaXRpYWxpemVyKFxuICAgICAgICBhcmdzLmtlcm5lbEluaXRpYWxpemVyIHx8IHRoaXMuREVGQVVMVF9LRVJORUxfSU5JVElBTElaRVIpO1xuICAgIHRoaXMucmVjdXJyZW50SW5pdGlhbGl6ZXIgPSBnZXRJbml0aWFsaXplcihcbiAgICAgICAgYXJncy5yZWN1cnJlbnRJbml0aWFsaXplciB8fCB0aGlzLkRFRkFVTFRfUkVDVVJSRU5UX0lOSVRJQUxJWkVSKTtcblxuICAgIHRoaXMuYmlhc0luaXRpYWxpemVyID1cbiAgICAgICAgZ2V0SW5pdGlhbGl6ZXIoYXJncy5iaWFzSW5pdGlhbGl6ZXIgfHwgdGhpcy5ERUZBVUxUX0JJQVNfSU5JVElBTElaRVIpO1xuXG4gICAgdGhpcy5rZXJuZWxSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3Mua2VybmVsUmVndWxhcml6ZXIpO1xuICAgIHRoaXMucmVjdXJyZW50UmVndWxhcml6ZXIgPSBnZXRSZWd1bGFyaXplcihhcmdzLnJlY3VycmVudFJlZ3VsYXJpemVyKTtcbiAgICB0aGlzLmJpYXNSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3MuYmlhc1JlZ3VsYXJpemVyKTtcblxuICAgIHRoaXMua2VybmVsQ29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5rZXJuZWxDb25zdHJhaW50KTtcbiAgICB0aGlzLnJlY3VycmVudENvbnN0cmFpbnQgPSBnZXRDb25zdHJhaW50KGFyZ3MucmVjdXJyZW50Q29uc3RyYWludCk7XG4gICAgdGhpcy5iaWFzQ29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5iaWFzQ29uc3RyYWludCk7XG5cbiAgICB0aGlzLmRyb3BvdXQgPSBtYXRoX3V0aWxzLm1pbihcbiAgICAgICAgWzEsIG1hdGhfdXRpbHMubWF4KFswLCBhcmdzLmRyb3BvdXQgPT0gbnVsbCA/IDAgOiBhcmdzLmRyb3BvdXRdKV0pO1xuICAgIHRoaXMucmVjdXJyZW50RHJvcG91dCA9IG1hdGhfdXRpbHMubWluKFtcbiAgICAgIDEsXG4gICAgICBtYXRoX3V0aWxzLm1heChcbiAgICAgICAgICBbMCwgYXJncy5yZWN1cnJlbnREcm9wb3V0ID09IG51bGwgPyAwIDogYXJncy5yZWN1cnJlbnREcm9wb3V0XSlcbiAgICBdKTtcbiAgICB0aGlzLmRyb3BvdXRGdW5jID0gYXJncy5kcm9wb3V0RnVuYztcbiAgICB0aGlzLmltcGxlbWVudGF0aW9uID0gYXJncy5pbXBsZW1lbnRhdGlvbjtcbiAgICB0aGlzLnN0YXRlU2l6ZSA9IHRoaXMudW5pdHM7XG4gICAgdGhpcy5kcm9wb3V0TWFzayA9IG51bGw7XG4gICAgdGhpcy5yZWN1cnJlbnREcm9wb3V0TWFzayA9IG51bGw7XG4gIH1cblxuICBwdWJsaWMgb3ZlcnJpZGUgYnVpbGQoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IHZvaWQge1xuICAgIGlucHV0U2hhcGUgPSBnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSk7XG4gICAgY29uc3QgaW5wdXREaW0gPSBpbnB1dFNoYXBlW2lucHV0U2hhcGUubGVuZ3RoIC0gMV07XG4gICAgdGhpcy5rZXJuZWwgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgJ2tlcm5lbCcsIFtpbnB1dERpbSwgdGhpcy51bml0cyAqIDNdLCBudWxsLCB0aGlzLmtlcm5lbEluaXRpYWxpemVyLFxuICAgICAgICB0aGlzLmtlcm5lbFJlZ3VsYXJpemVyLCB0cnVlLCB0aGlzLmtlcm5lbENvbnN0cmFpbnQpO1xuICAgIHRoaXMucmVjdXJyZW50S2VybmVsID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICdyZWN1cnJlbnRfa2VybmVsJywgW3RoaXMudW5pdHMsIHRoaXMudW5pdHMgKiAzXSwgbnVsbCxcbiAgICAgICAgdGhpcy5yZWN1cnJlbnRJbml0aWFsaXplciwgdGhpcy5yZWN1cnJlbnRSZWd1bGFyaXplciwgdHJ1ZSxcbiAgICAgICAgdGhpcy5yZWN1cnJlbnRDb25zdHJhaW50KTtcbiAgICBpZiAodGhpcy51c2VCaWFzKSB7XG4gICAgICB0aGlzLmJpYXMgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgICAnYmlhcycsIFt0aGlzLnVuaXRzICogM10sIG51bGwsIHRoaXMuYmlhc0luaXRpYWxpemVyLFxuICAgICAgICAgIHRoaXMuYmlhc1JlZ3VsYXJpemVyLCB0cnVlLCB0aGlzLmJpYXNDb25zdHJhaW50KTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5iaWFzID0gbnVsbDtcbiAgICB9XG4gICAgLy8gUG9ydGluZyBOb3RlczogVW5saWtlIHRoZSBQeUtlcmFzIGltcGxlbWVudGF0aW9uLCB3ZSBwZXJmb3JtIHNsaWNpbmdcbiAgICAvLyAgIG9mIHRoZSB3ZWlnaHRzIGFuZCBiaWFzIGluIHRoZSBjYWxsKCkgbWV0aG9kLCBhdCBleGVjdXRpb24gdGltZS5cbiAgICB0aGlzLmJ1aWx0ID0gdHJ1ZTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpbnB1dHMgPSBpbnB1dHMgYXMgVGVuc29yW107XG4gICAgICBpZiAoaW5wdXRzLmxlbmd0aCAhPT0gMikge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBHUlVDZWxsIGV4cGVjdHMgMiBpbnB1dCBUZW5zb3JzIChpbnB1dHMsIGgsIGMpLCBnb3QgYCArXG4gICAgICAgICAgICBgJHtpbnB1dHMubGVuZ3RofS5gKTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgdHJhaW5pbmcgPSBrd2FyZ3NbJ3RyYWluaW5nJ10gPT0gbnVsbCA/IGZhbHNlIDoga3dhcmdzWyd0cmFpbmluZyddO1xuICAgICAgbGV0IGhUTWludXMxID0gaW5wdXRzWzFdOyAgLy8gUHJldmlvdXMgbWVtb3J5IHN0YXRlLlxuICAgICAgaW5wdXRzID0gaW5wdXRzWzBdO1xuXG4gICAgICAvLyBOb3RlOiBGb3Igc3VwZXJpb3IgcGVyZm9ybWFuY2UsIFRlbnNvckZsb3cuanMgYWx3YXlzIHVzZXNcbiAgICAgIC8vIGltcGxlbWVudGF0aW9uIDIsIHJlZ2FyZGxlc3Mgb2YgdGhlIGFjdHVhbCB2YWx1ZSBvZlxuICAgICAgLy8gY29uZmlnLmltcGxlbWVudGF0aW9uLlxuICAgICAgaWYgKDAgPCB0aGlzLmRyb3BvdXQgJiYgdGhpcy5kcm9wb3V0IDwgMSAmJiB0aGlzLmRyb3BvdXRNYXNrID09IG51bGwpIHtcbiAgICAgICAgdGhpcy5kcm9wb3V0TWFzayA9IGdlbmVyYXRlRHJvcG91dE1hc2soe1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICBvbmVzOiAoKSA9PiB0ZmMub25lc0xpa2UoaW5wdXRzIGFzIFRlbnNvciksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJhdGU6IHRoaXMuZHJvcG91dCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdHJhaW5pbmcsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvdW50OiAzLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkcm9wb3V0RnVuYzogdGhpcy5kcm9wb3V0RnVuYyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pIGFzIFRlbnNvcltdO1xuICAgICAgfVxuICAgICAgaWYgKDAgPCB0aGlzLnJlY3VycmVudERyb3BvdXQgJiYgdGhpcy5yZWN1cnJlbnREcm9wb3V0IDwgMSAmJlxuICAgICAgICAgIHRoaXMucmVjdXJyZW50RHJvcG91dE1hc2sgPT0gbnVsbCkge1xuICAgICAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrID0gZ2VuZXJhdGVEcm9wb3V0TWFzayh7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9uZXM6ICgpID0+IHRmYy5vbmVzTGlrZShoVE1pbnVzMSksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJhdGU6IHRoaXMucmVjdXJyZW50RHJvcG91dCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdHJhaW5pbmcsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvdW50OiAzLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkcm9wb3V0RnVuYzogdGhpcy5kcm9wb3V0RnVuYyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pIGFzIFRlbnNvcltdO1xuICAgICAgfVxuICAgICAgY29uc3QgZHBNYXNrID0gdGhpcy5kcm9wb3V0TWFzayBhcyBbVGVuc29yLCBUZW5zb3IsIFRlbnNvcl07XG4gICAgICBjb25zdCByZWNEcE1hc2sgPSB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrIGFzIFtUZW5zb3IsIFRlbnNvciwgVGVuc29yXTtcbiAgICAgIGxldCB6OiBUZW5zb3I7XG4gICAgICBsZXQgcjogVGVuc29yO1xuICAgICAgbGV0IGhoOiBUZW5zb3I7XG5cbiAgICAgIGlmICgwIDwgdGhpcy5kcm9wb3V0ICYmIHRoaXMuZHJvcG91dCA8IDEpIHtcbiAgICAgICAgaW5wdXRzID0gdGZjLm11bChpbnB1dHMsIGRwTWFza1swXSk7XG4gICAgICB9XG4gICAgICBsZXQgbWF0cml4WCA9IEsuZG90KGlucHV0cywgdGhpcy5rZXJuZWwucmVhZCgpKTtcbiAgICAgIGlmICh0aGlzLnVzZUJpYXMpIHtcbiAgICAgICAgbWF0cml4WCA9IEsuYmlhc0FkZChtYXRyaXhYLCB0aGlzLmJpYXMucmVhZCgpKTtcbiAgICAgIH1cbiAgICAgIGlmICgwIDwgdGhpcy5yZWN1cnJlbnREcm9wb3V0ICYmIHRoaXMucmVjdXJyZW50RHJvcG91dCA8IDEpIHtcbiAgICAgICAgaFRNaW51czEgPSB0ZmMubXVsKGhUTWludXMxLCByZWNEcE1hc2tbMF0pO1xuICAgICAgfVxuXG4gICAgICBjb25zdCByZWN1cnJlbnRLZXJuZWxWYWx1ZSA9IHRoaXMucmVjdXJyZW50S2VybmVsLnJlYWQoKTtcbiAgICAgIGNvbnN0IFtyazEsIHJrMl0gPSB0ZmMuc3BsaXQoXG4gICAgICAgICAgcmVjdXJyZW50S2VybmVsVmFsdWUsIFsyICogdGhpcy51bml0cywgdGhpcy51bml0c10sXG4gICAgICAgICAgcmVjdXJyZW50S2VybmVsVmFsdWUucmFuayAtIDEpO1xuICAgICAgY29uc3QgbWF0cml4SW5uZXIgPSBLLmRvdChoVE1pbnVzMSwgcmsxKTtcblxuICAgICAgY29uc3QgW3haLCB4UiwgeEhdID0gdGZjLnNwbGl0KG1hdHJpeFgsIDMsIG1hdHJpeFgucmFuayAtIDEpO1xuICAgICAgY29uc3QgW3JlY3VycmVudFosIHJlY3VycmVudFJdID1cbiAgICAgICAgICB0ZmMuc3BsaXQobWF0cml4SW5uZXIsIDIsIG1hdHJpeElubmVyLnJhbmsgLSAxKTtcbiAgICAgIHogPSB0aGlzLnJlY3VycmVudEFjdGl2YXRpb24uYXBwbHkodGZjLmFkZCh4WiwgcmVjdXJyZW50WikpO1xuICAgICAgciA9IHRoaXMucmVjdXJyZW50QWN0aXZhdGlvbi5hcHBseSh0ZmMuYWRkKHhSLCByZWN1cnJlbnRSKSk7XG5cbiAgICAgIGNvbnN0IHJlY3VycmVudEggPSBLLmRvdCh0ZmMubXVsKHIsIGhUTWludXMxKSwgcmsyKTtcbiAgICAgIGhoID0gdGhpcy5hY3RpdmF0aW9uLmFwcGx5KHRmYy5hZGQoeEgsIHJlY3VycmVudEgpKTtcblxuICAgICAgY29uc3QgaCA9XG4gICAgICAgICAgdGZjLmFkZCh0ZmMubXVsKHosIGhUTWludXMxKSwgdGZjLm11bCh0ZmMuYWRkKDEsIHRmYy5uZWcoeikpLCBoaCkpO1xuICAgICAgLy8gVE9ETyhjYWlzKTogQWRkIHVzZV9sZWFybmluZ19waGFzZSBmbGFnIHByb3Blcmx5LlxuICAgICAgcmV0dXJuIFtoLCBoXTtcbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcblxuICAgIGNvbnN0IGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge1xuICAgICAgdW5pdHM6IHRoaXMudW5pdHMsXG4gICAgICBhY3RpdmF0aW9uOiBzZXJpYWxpemVBY3RpdmF0aW9uKHRoaXMuYWN0aXZhdGlvbiksXG4gICAgICByZWN1cnJlbnRBY3RpdmF0aW9uOiBzZXJpYWxpemVBY3RpdmF0aW9uKHRoaXMucmVjdXJyZW50QWN0aXZhdGlvbiksXG4gICAgICB1c2VCaWFzOiB0aGlzLnVzZUJpYXMsXG4gICAgICBrZXJuZWxJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5rZXJuZWxJbml0aWFsaXplciksXG4gICAgICByZWN1cnJlbnRJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5yZWN1cnJlbnRJbml0aWFsaXplciksXG4gICAgICBiaWFzSW5pdGlhbGl6ZXI6IHNlcmlhbGl6ZUluaXRpYWxpemVyKHRoaXMuYmlhc0luaXRpYWxpemVyKSxcbiAgICAgIGtlcm5lbFJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLmtlcm5lbFJlZ3VsYXJpemVyKSxcbiAgICAgIHJlY3VycmVudFJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLnJlY3VycmVudFJlZ3VsYXJpemVyKSxcbiAgICAgIGJpYXNSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5iaWFzUmVndWxhcml6ZXIpLFxuICAgICAgYWN0aXZpdHlSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5hY3Rpdml0eVJlZ3VsYXJpemVyKSxcbiAgICAgIGtlcm5lbENvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5rZXJuZWxDb25zdHJhaW50KSxcbiAgICAgIHJlY3VycmVudENvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5yZWN1cnJlbnRDb25zdHJhaW50KSxcbiAgICAgIGJpYXNDb25zdHJhaW50OiBzZXJpYWxpemVDb25zdHJhaW50KHRoaXMuYmlhc0NvbnN0cmFpbnQpLFxuICAgICAgZHJvcG91dDogdGhpcy5kcm9wb3V0LFxuICAgICAgcmVjdXJyZW50RHJvcG91dDogdGhpcy5yZWN1cnJlbnREcm9wb3V0LFxuICAgICAgaW1wbGVtZW50YXRpb246IHRoaXMuaW1wbGVtZW50YXRpb24sXG4gICAgICByZXNldEFmdGVyOiBmYWxzZVxuICAgIH07XG5cbiAgICByZXR1cm4gey4uLmJhc2VDb25maWcsIC4uLmNvbmZpZ307XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhHUlVDZWxsKTtcblxuLy8gUG9ydGluZyBOb3RlOiBTaW5jZSB0aGlzIGlzIGEgc3VwZXJzZXQgb2YgU2ltcGxlUk5OTGF5ZXJDb25maWcsIHdlIGluaGVyaXRcbi8vICAgZnJvbSB0aGF0IGludGVyZmFjZSBpbnN0ZWFkIG9mIHJlcGVhdGluZyB0aGUgZmllbGRzIGhlcmUuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgR1JVTGF5ZXJBcmdzIGV4dGVuZHMgU2ltcGxlUk5OTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIEFjdGl2YXRpb24gZnVuY3Rpb24gdG8gdXNlIGZvciB0aGUgcmVjdXJyZW50IHN0ZXAuXG4gICAqXG4gICAqIERlZmF1bHRzIHRvIGhhcmQgc2lnbW9pZCAoYGhhcmRTaWdtb2lkYCkuXG4gICAqXG4gICAqIElmIGBudWxsYCwgbm8gYWN0aXZhdGlvbiBpcyBhcHBsaWVkLlxuICAgKi9cbiAgcmVjdXJyZW50QWN0aXZhdGlvbj86IEFjdGl2YXRpb25JZGVudGlmaWVyO1xuXG4gIC8qKlxuICAgKiBJbXBsZW1lbnRhdGlvbiBtb2RlLCBlaXRoZXIgMSBvciAyLlxuICAgKlxuICAgKiBNb2RlIDEgd2lsbCBzdHJ1Y3R1cmUgaXRzIG9wZXJhdGlvbnMgYXMgYSBsYXJnZXIgbnVtYmVyIG9mXG4gICAqIHNtYWxsZXIgZG90IHByb2R1Y3RzIGFuZCBhZGRpdGlvbnMuXG4gICAqXG4gICAqIE1vZGUgMiB3aWxsIGJhdGNoIHRoZW0gaW50byBmZXdlciwgbGFyZ2VyIG9wZXJhdGlvbnMuIFRoZXNlIG1vZGVzIHdpbGxcbiAgICogaGF2ZSBkaWZmZXJlbnQgcGVyZm9ybWFuY2UgcHJvZmlsZXMgb24gZGlmZmVyZW50IGhhcmR3YXJlIGFuZFxuICAgKiBmb3IgZGlmZmVyZW50IGFwcGxpY2F0aW9ucy5cbiAgICpcbiAgICogTm90ZTogRm9yIHN1cGVyaW9yIHBlcmZvcm1hbmNlLCBUZW5zb3JGbG93LmpzIGFsd2F5cyB1c2VzIGltcGxlbWVudGF0aW9uXG4gICAqIDIsIHJlZ2FyZGxlc3Mgb2YgdGhlIGFjdHVhbCB2YWx1ZSBvZiB0aGlzIGNvbmZpZ3VyYXRpb24gZmllbGQuXG4gICAqL1xuICBpbXBsZW1lbnRhdGlvbj86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIEdSVSBleHRlbmRzIFJOTiB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgb3ZlcnJpZGUgY2xhc3NOYW1lID0gJ0dSVSc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IEdSVUxheWVyQXJncykge1xuICAgIGlmIChhcmdzLmltcGxlbWVudGF0aW9uID09PSAwKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgJ2BpbXBsZW1lbnRhdGlvbj0wYCBoYXMgYmVlbiBkZXByZWNhdGVkLCBhbmQgbm93IGRlZmF1bHRzIHRvICcgK1xuICAgICAgICAgICdgaW1wbGVtZW50YXRpb249MWAuIFBsZWFzZSB1cGRhdGUgeW91ciBsYXllciBjYWxsLicpO1xuICAgIH1cbiAgICBhcmdzLmNlbGwgPSBuZXcgR1JVQ2VsbChhcmdzKTtcbiAgICBzdXBlcihhcmdzIGFzIFJOTkxheWVyQXJncyk7XG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIGFjdGl2aXR5UmVndWxhcml6ZXIuXG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgaWYgKHRoaXMuY2VsbC5kcm9wb3V0TWFzayAhPSBudWxsKSB7XG4gICAgICAgIHRmYy5kaXNwb3NlKHRoaXMuY2VsbC5kcm9wb3V0TWFzayk7XG4gICAgICAgIHRoaXMuY2VsbC5kcm9wb3V0TWFzayA9IG51bGw7XG4gICAgICB9XG4gICAgICBpZiAodGhpcy5jZWxsLnJlY3VycmVudERyb3BvdXRNYXNrICE9IG51bGwpIHtcbiAgICAgICAgdGZjLmRpc3Bvc2UodGhpcy5jZWxsLnJlY3VycmVudERyb3BvdXRNYXNrKTtcbiAgICAgICAgdGhpcy5jZWxsLnJlY3VycmVudERyb3BvdXRNYXNrID0gbnVsbDtcbiAgICAgIH1cbiAgICAgIGNvbnN0IG1hc2sgPSBrd2FyZ3MgPT0gbnVsbCA/IG51bGwgOiBrd2FyZ3NbJ21hc2snXTtcbiAgICAgIGNvbnN0IHRyYWluaW5nID0ga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWyd0cmFpbmluZyddO1xuICAgICAgY29uc3QgaW5pdGlhbFN0YXRlOiBUZW5zb3JbXSA9XG4gICAgICAgICAga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydpbml0aWFsU3RhdGUnXTtcbiAgICAgIHJldHVybiBzdXBlci5jYWxsKGlucHV0cywge21hc2ssIHRyYWluaW5nLCBpbml0aWFsU3RhdGV9KTtcbiAgICB9KTtcbiAgfVxuXG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgb3ZlcnJpZGUgZnJvbUNvbmZpZzxUIGV4dGVuZHMgc2VyaWFsaXphdGlvbi5TZXJpYWxpemFibGU+KFxuICAgICAgY2xzOiBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZUNvbnN0cnVjdG9yPFQ+LFxuICAgICAgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QpOiBUIHtcbiAgICBpZiAoY29uZmlnWydpbXBsbWVudGF0aW9uJ10gPT09IDApIHtcbiAgICAgIGNvbmZpZ1snaW1wbGVtZW50YXRpb24nXSA9IDE7XG4gICAgfVxuICAgIHJldHVybiBuZXcgY2xzKGNvbmZpZyk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhHUlUpO1xuXG4vLyBQb3J0aW5nIE5vdGU6IFNpbmNlIHRoaXMgaXMgYSBzdXBlcnNldCBvZiBTaW1wbGVSTk5MYXllckNvbmZpZywgd2UgZXh0ZW5kXG4vLyAgIHRoYXQgaW50ZXJmYWNlIGluc3RlYWQgb2YgcmVwZWF0aW5nIHRoZSBmaWVsZHMuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgTFNUTUNlbGxMYXllckFyZ3MgZXh0ZW5kcyBTaW1wbGVSTk5DZWxsTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIEFjdGl2YXRpb24gZnVuY3Rpb24gdG8gdXNlIGZvciB0aGUgcmVjdXJyZW50IHN0ZXAuXG4gICAqXG4gICAqIERlZmF1bHRzIHRvIGhhcmQgc2lnbW9pZCAoYGhhcmRTaWdtb2lkYCkuXG4gICAqXG4gICAqIElmIGBudWxsYCwgbm8gYWN0aXZhdGlvbiBpcyBhcHBsaWVkLlxuICAgKi9cbiAgcmVjdXJyZW50QWN0aXZhdGlvbj86IEFjdGl2YXRpb25JZGVudGlmaWVyO1xuXG4gIC8qKlxuICAgKiBJZiBgdHJ1ZWAsIGFkZCAxIHRvIHRoZSBiaWFzIG9mIHRoZSBmb3JnZXQgZ2F0ZSBhdCBpbml0aWFsaXphdGlvbi5cbiAgICogU2V0dGluZyBpdCB0byBgdHJ1ZWAgd2lsbCBhbHNvIGZvcmNlIGBiaWFzSW5pdGlhbGl6ZXIgPSAnemVyb3MnYC5cbiAgICogVGhpcyBpcyByZWNvbW1lbmRlZCBpblxuICAgKiBbSm96ZWZvd2ljeiBldFxuICAgKiBhbC5dKGh0dHA6Ly93d3cuam1sci5vcmcvcHJvY2VlZGluZ3MvcGFwZXJzL3YzNy9qb3plZm93aWN6MTUucGRmKVxuICAgKi9cbiAgdW5pdEZvcmdldEJpYXM/OiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBJbXBsZW1lbnRhdGlvbiBtb2RlLCBlaXRoZXIgMSBvciAyLlxuICAgKlxuICAgKiBNb2RlIDEgd2lsbCBzdHJ1Y3R1cmUgaXRzIG9wZXJhdGlvbnMgYXMgYSBsYXJnZXIgbnVtYmVyIG9mXG4gICAqICAgc21hbGxlciBkb3QgcHJvZHVjdHMgYW5kIGFkZGl0aW9ucy5cbiAgICpcbiAgICogTW9kZSAyIHdpbGwgYmF0Y2ggdGhlbSBpbnRvIGZld2VyLCBsYXJnZXIgb3BlcmF0aW9ucy4gVGhlc2UgbW9kZXMgd2lsbFxuICAgKiBoYXZlIGRpZmZlcmVudCBwZXJmb3JtYW5jZSBwcm9maWxlcyBvbiBkaWZmZXJlbnQgaGFyZHdhcmUgYW5kXG4gICAqIGZvciBkaWZmZXJlbnQgYXBwbGljYXRpb25zLlxuICAgKlxuICAgKiBOb3RlOiBGb3Igc3VwZXJpb3IgcGVyZm9ybWFuY2UsIFRlbnNvckZsb3cuanMgYWx3YXlzIHVzZXMgaW1wbGVtZW50YXRpb25cbiAgICogMiwgcmVnYXJkbGVzcyBvZiB0aGUgYWN0dWFsIHZhbHVlIG9mIHRoaXMgY29uZmlndXJhdGlvbiBmaWVsZC5cbiAgICovXG4gIGltcGxlbWVudGF0aW9uPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgTFNUTUNlbGwgZXh0ZW5kcyBSTk5DZWxsIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnTFNUTUNlbGwnO1xuICByZWFkb25seSB1bml0czogbnVtYmVyO1xuICByZWFkb25seSBhY3RpdmF0aW9uOiBBY3RpdmF0aW9uO1xuICByZWFkb25seSByZWN1cnJlbnRBY3RpdmF0aW9uOiBBY3RpdmF0aW9uO1xuICByZWFkb25seSB1c2VCaWFzOiBib29sZWFuO1xuXG4gIHJlYWRvbmx5IGtlcm5lbEluaXRpYWxpemVyOiBJbml0aWFsaXplcjtcbiAgcmVhZG9ubHkgcmVjdXJyZW50SW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICByZWFkb25seSBiaWFzSW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICByZWFkb25seSB1bml0Rm9yZ2V0QmlhczogYm9vbGVhbjtcblxuICByZWFkb25seSBrZXJuZWxDb25zdHJhaW50OiBDb25zdHJhaW50O1xuICByZWFkb25seSByZWN1cnJlbnRDb25zdHJhaW50OiBDb25zdHJhaW50O1xuICByZWFkb25seSBiaWFzQ29uc3RyYWludDogQ29uc3RyYWludDtcblxuICByZWFkb25seSBrZXJuZWxSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudFJlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcbiAgcmVhZG9ubHkgYmlhc1JlZ3VsYXJpemVyOiBSZWd1bGFyaXplcjtcblxuICByZWFkb25seSBkcm9wb3V0OiBudW1iZXI7XG4gIHJlYWRvbmx5IHJlY3VycmVudERyb3BvdXQ6IG51bWJlcjtcbiAgcmVhZG9ubHkgZHJvcG91dEZ1bmM6IEZ1bmN0aW9uO1xuXG4gIHJlYWRvbmx5IHN0YXRlU2l6ZTogbnVtYmVyW107XG4gIHJlYWRvbmx5IGltcGxlbWVudGF0aW9uOiBudW1iZXI7XG5cbiAgcmVhZG9ubHkgREVGQVVMVF9BQ1RJVkFUSU9OID0gJ3RhbmgnO1xuICByZWFkb25seSBERUZBVUxUX1JFQ1VSUkVOVF9BQ1RJVkFUSU9OID0gJ2hhcmRTaWdtb2lkJztcbiAgcmVhZG9ubHkgREVGQVVMVF9LRVJORUxfSU5JVElBTElaRVIgPSAnZ2xvcm90Tm9ybWFsJztcbiAgcmVhZG9ubHkgREVGQVVMVF9SRUNVUlJFTlRfSU5JVElBTElaRVIgPSAnb3J0aG9nb25hbCc7XG5cbiAgcmVhZG9ubHkgREVGQVVMVF9CSUFTX0lOSVRJQUxJWkVSID0gJ3plcm9zJztcblxuICBrZXJuZWw6IExheWVyVmFyaWFibGU7XG4gIHJlY3VycmVudEtlcm5lbDogTGF5ZXJWYXJpYWJsZTtcbiAgYmlhczogTGF5ZXJWYXJpYWJsZTtcblxuICBjb25zdHJ1Y3RvcihhcmdzOiBMU1RNQ2VsbExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuXG4gICAgdGhpcy51bml0cyA9IGFyZ3MudW5pdHM7XG4gICAgYXNzZXJ0UG9zaXRpdmVJbnRlZ2VyKHRoaXMudW5pdHMsICd1bml0cycpO1xuICAgIHRoaXMuYWN0aXZhdGlvbiA9IGdldEFjdGl2YXRpb24oXG4gICAgICAgIGFyZ3MuYWN0aXZhdGlvbiA9PT0gdW5kZWZpbmVkID8gdGhpcy5ERUZBVUxUX0FDVElWQVRJT04gOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFyZ3MuYWN0aXZhdGlvbik7XG4gICAgdGhpcy5yZWN1cnJlbnRBY3RpdmF0aW9uID0gZ2V0QWN0aXZhdGlvbihcbiAgICAgICAgYXJncy5yZWN1cnJlbnRBY3RpdmF0aW9uID09PSB1bmRlZmluZWQgP1xuICAgICAgICAgICAgdGhpcy5ERUZBVUxUX1JFQ1VSUkVOVF9BQ1RJVkFUSU9OIDpcbiAgICAgICAgICAgIGFyZ3MucmVjdXJyZW50QWN0aXZhdGlvbik7XG4gICAgdGhpcy51c2VCaWFzID0gYXJncy51c2VCaWFzID09IG51bGwgPyB0cnVlIDogYXJncy51c2VCaWFzO1xuXG4gICAgdGhpcy5rZXJuZWxJbml0aWFsaXplciA9IGdldEluaXRpYWxpemVyKFxuICAgICAgICBhcmdzLmtlcm5lbEluaXRpYWxpemVyIHx8IHRoaXMuREVGQVVMVF9LRVJORUxfSU5JVElBTElaRVIpO1xuICAgIHRoaXMucmVjdXJyZW50SW5pdGlhbGl6ZXIgPSBnZXRJbml0aWFsaXplcihcbiAgICAgICAgYXJncy5yZWN1cnJlbnRJbml0aWFsaXplciB8fCB0aGlzLkRFRkFVTFRfUkVDVVJSRU5UX0lOSVRJQUxJWkVSKTtcblxuICAgIHRoaXMuYmlhc0luaXRpYWxpemVyID1cbiAgICAgICAgZ2V0SW5pdGlhbGl6ZXIoYXJncy5iaWFzSW5pdGlhbGl6ZXIgfHwgdGhpcy5ERUZBVUxUX0JJQVNfSU5JVElBTElaRVIpO1xuICAgIHRoaXMudW5pdEZvcmdldEJpYXMgPSBhcmdzLnVuaXRGb3JnZXRCaWFzO1xuXG4gICAgdGhpcy5rZXJuZWxSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3Mua2VybmVsUmVndWxhcml6ZXIpO1xuICAgIHRoaXMucmVjdXJyZW50UmVndWxhcml6ZXIgPSBnZXRSZWd1bGFyaXplcihhcmdzLnJlY3VycmVudFJlZ3VsYXJpemVyKTtcbiAgICB0aGlzLmJpYXNSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3MuYmlhc1JlZ3VsYXJpemVyKTtcblxuICAgIHRoaXMua2VybmVsQ29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5rZXJuZWxDb25zdHJhaW50KTtcbiAgICB0aGlzLnJlY3VycmVudENvbnN0cmFpbnQgPSBnZXRDb25zdHJhaW50KGFyZ3MucmVjdXJyZW50Q29uc3RyYWludCk7XG4gICAgdGhpcy5iaWFzQ29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5iaWFzQ29uc3RyYWludCk7XG5cbiAgICB0aGlzLmRyb3BvdXQgPSBtYXRoX3V0aWxzLm1pbihcbiAgICAgICAgWzEsIG1hdGhfdXRpbHMubWF4KFswLCBhcmdzLmRyb3BvdXQgPT0gbnVsbCA/IDAgOiBhcmdzLmRyb3BvdXRdKV0pO1xuICAgIHRoaXMucmVjdXJyZW50RHJvcG91dCA9IG1hdGhfdXRpbHMubWluKFtcbiAgICAgIDEsXG4gICAgICBtYXRoX3V0aWxzLm1heChcbiAgICAgICAgICBbMCwgYXJncy5yZWN1cnJlbnREcm9wb3V0ID09IG51bGwgPyAwIDogYXJncy5yZWN1cnJlbnREcm9wb3V0XSlcbiAgICBdKTtcbiAgICB0aGlzLmRyb3BvdXRGdW5jID0gYXJncy5kcm9wb3V0RnVuYztcbiAgICB0aGlzLmltcGxlbWVudGF0aW9uID0gYXJncy5pbXBsZW1lbnRhdGlvbjtcbiAgICB0aGlzLnN0YXRlU2l6ZSA9IFt0aGlzLnVuaXRzLCB0aGlzLnVuaXRzXTtcbiAgICB0aGlzLmRyb3BvdXRNYXNrID0gbnVsbDtcbiAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrID0gbnVsbDtcbiAgfVxuXG4gIHB1YmxpYyBvdmVycmlkZSBidWlsZChpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogdm9pZCB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBjb25zdCBpbnB1dERpbSA9IGlucHV0U2hhcGVbaW5wdXRTaGFwZS5sZW5ndGggLSAxXTtcbiAgICB0aGlzLmtlcm5lbCA9IHRoaXMuYWRkV2VpZ2h0KFxuICAgICAgICAna2VybmVsJywgW2lucHV0RGltLCB0aGlzLnVuaXRzICogNF0sIG51bGwsIHRoaXMua2VybmVsSW5pdGlhbGl6ZXIsXG4gICAgICAgIHRoaXMua2VybmVsUmVndWxhcml6ZXIsIHRydWUsIHRoaXMua2VybmVsQ29uc3RyYWludCk7XG4gICAgdGhpcy5yZWN1cnJlbnRLZXJuZWwgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgJ3JlY3VycmVudF9rZXJuZWwnLCBbdGhpcy51bml0cywgdGhpcy51bml0cyAqIDRdLCBudWxsLFxuICAgICAgICB0aGlzLnJlY3VycmVudEluaXRpYWxpemVyLCB0aGlzLnJlY3VycmVudFJlZ3VsYXJpemVyLCB0cnVlLFxuICAgICAgICB0aGlzLnJlY3VycmVudENvbnN0cmFpbnQpO1xuICAgIGxldCBiaWFzSW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICAgIGlmICh0aGlzLnVzZUJpYXMpIHtcbiAgICAgIGlmICh0aGlzLnVuaXRGb3JnZXRCaWFzKSB7XG4gICAgICAgIGNvbnN0IGNhcHR1cmVkQmlhc0luaXQgPSB0aGlzLmJpYXNJbml0aWFsaXplcjtcbiAgICAgICAgY29uc3QgY2FwdHVyZWRVbml0cyA9IHRoaXMudW5pdHM7XG4gICAgICAgIGJpYXNJbml0aWFsaXplciA9IG5ldyAoY2xhc3MgQ3VzdG9tSW5pdCBleHRlbmRzIEluaXRpYWxpemVyIHtcbiAgICAgICAgICAvKiogQG5vY29sbGFwc2UgKi9cbiAgICAgICAgICBzdGF0aWMgY2xhc3NOYW1lID0gJ0N1c3RvbUluaXQnO1xuXG4gICAgICAgICAgYXBwbHkoc2hhcGU6IFNoYXBlLCBkdHlwZT86IERhdGFUeXBlKTogVGVuc29yIHtcbiAgICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IE1vcmUgaW5mb3JtYXRpdmUgdmFyaWFibGUgbmFtZXM/XG4gICAgICAgICAgICBjb25zdCBiSSA9IGNhcHR1cmVkQmlhc0luaXQuYXBwbHkoW2NhcHR1cmVkVW5pdHNdKTtcbiAgICAgICAgICAgIGNvbnN0IGJGID0gKG5ldyBPbmVzKCkpLmFwcGx5KFtjYXB0dXJlZFVuaXRzXSk7XG4gICAgICAgICAgICBjb25zdCBiQ0FuZEggPSBjYXB0dXJlZEJpYXNJbml0LmFwcGx5KFtjYXB0dXJlZFVuaXRzICogMl0pO1xuICAgICAgICAgICAgcmV0dXJuIEsuY29uY2F0QWxvbmdGaXJzdEF4aXMoXG4gICAgICAgICAgICAgICAgSy5jb25jYXRBbG9uZ0ZpcnN0QXhpcyhiSSwgYkYpLCBiQ0FuZEgpO1xuICAgICAgICAgIH1cbiAgICAgICAgfSkoKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGJpYXNJbml0aWFsaXplciA9IHRoaXMuYmlhc0luaXRpYWxpemVyO1xuICAgICAgfVxuICAgICAgdGhpcy5iaWFzID0gdGhpcy5hZGRXZWlnaHQoXG4gICAgICAgICAgJ2JpYXMnLCBbdGhpcy51bml0cyAqIDRdLCBudWxsLCBiaWFzSW5pdGlhbGl6ZXIsIHRoaXMuYmlhc1JlZ3VsYXJpemVyLFxuICAgICAgICAgIHRydWUsIHRoaXMuYmlhc0NvbnN0cmFpbnQpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmJpYXMgPSBudWxsO1xuICAgIH1cbiAgICAvLyBQb3J0aW5nIE5vdGVzOiBVbmxpa2UgdGhlIFB5S2VyYXMgaW1wbGVtZW50YXRpb24sIHdlIHBlcmZvcm0gc2xpY2luZ1xuICAgIC8vICAgb2YgdGhlIHdlaWdodHMgYW5kIGJpYXMgaW4gdGhlIGNhbGwoKSBtZXRob2QsIGF0IGV4ZWN1dGlvbiB0aW1lLlxuICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuICB9XG5cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IHRyYWluaW5nID0ga3dhcmdzWyd0cmFpbmluZyddID09IG51bGwgPyBmYWxzZSA6IGt3YXJnc1sndHJhaW5pbmcnXTtcbiAgICAgIGlucHV0cyA9IGlucHV0cyBhcyBUZW5zb3JbXTtcbiAgICAgIGlmIChpbnB1dHMubGVuZ3RoICE9PSAzKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYExTVE1DZWxsIGV4cGVjdHMgMyBpbnB1dCBUZW5zb3JzIChpbnB1dHMsIGgsIGMpLCBnb3QgYCArXG4gICAgICAgICAgICBgJHtpbnB1dHMubGVuZ3RofS5gKTtcbiAgICAgIH1cbiAgICAgIGxldCBoVE1pbnVzMSA9IGlucHV0c1sxXTsgICAgLy8gUHJldmlvdXMgbWVtb3J5IHN0YXRlLlxuICAgICAgY29uc3QgY1RNaW51czEgPSBpbnB1dHNbMl07ICAvLyBQcmV2aW91cyBjYXJyeSBzdGF0ZS5cbiAgICAgIGlucHV0cyA9IGlucHV0c1swXTtcbiAgICAgIGlmICgwIDwgdGhpcy5kcm9wb3V0ICYmIHRoaXMuZHJvcG91dCA8IDEgJiYgdGhpcy5kcm9wb3V0TWFzayA9PSBudWxsKSB7XG4gICAgICAgIHRoaXMuZHJvcG91dE1hc2sgPSBnZW5lcmF0ZURyb3BvdXRNYXNrKHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgb25lczogKCkgPT4gdGZjLm9uZXNMaWtlKGlucHV0cyBhcyBUZW5zb3IpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICByYXRlOiB0aGlzLmRyb3BvdXQsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRyYWluaW5nLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb3VudDogNCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZHJvcG91dEZ1bmM6IHRoaXMuZHJvcG91dEZ1bmNcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pIGFzIFRlbnNvcltdO1xuICAgICAgfVxuICAgICAgaWYgKDAgPCB0aGlzLnJlY3VycmVudERyb3BvdXQgJiYgdGhpcy5yZWN1cnJlbnREcm9wb3V0IDwgMSAmJlxuICAgICAgICAgIHRoaXMucmVjdXJyZW50RHJvcG91dE1hc2sgPT0gbnVsbCkge1xuICAgICAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrID0gZ2VuZXJhdGVEcm9wb3V0TWFzayh7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9uZXM6ICgpID0+IHRmYy5vbmVzTGlrZShoVE1pbnVzMSksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJhdGU6IHRoaXMucmVjdXJyZW50RHJvcG91dCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdHJhaW5pbmcsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvdW50OiA0LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkcm9wb3V0RnVuYzogdGhpcy5kcm9wb3V0RnVuY1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSkgYXMgVGVuc29yW107XG4gICAgICB9XG4gICAgICBjb25zdCBkcE1hc2sgPSB0aGlzLmRyb3BvdXRNYXNrIGFzIFtUZW5zb3IsIFRlbnNvciwgVGVuc29yLCBUZW5zb3JdO1xuICAgICAgY29uc3QgcmVjRHBNYXNrID1cbiAgICAgICAgICB0aGlzLnJlY3VycmVudERyb3BvdXRNYXNrIGFzIFtUZW5zb3IsIFRlbnNvciwgVGVuc29yLCBUZW5zb3JdO1xuXG4gICAgICAvLyBOb3RlOiBGb3Igc3VwZXJpb3IgcGVyZm9ybWFuY2UsIFRlbnNvckZsb3cuanMgYWx3YXlzIHVzZXNcbiAgICAgIC8vIGltcGxlbWVudGF0aW9uIDIgcmVnYXJkbGVzcyBvZiB0aGUgYWN0dWFsIHZhbHVlIG9mXG4gICAgICAvLyBjb25maWcuaW1wbGVtZW50YXRpb24uXG4gICAgICBsZXQgaTogVGVuc29yO1xuICAgICAgbGV0IGY6IFRlbnNvcjtcbiAgICAgIGxldCBjOiBUZW5zb3I7XG4gICAgICBsZXQgbzogVGVuc29yO1xuICAgICAgaWYgKDAgPCB0aGlzLmRyb3BvdXQgJiYgdGhpcy5kcm9wb3V0IDwgMSkge1xuICAgICAgICBpbnB1dHMgPSB0ZmMubXVsKGlucHV0cywgZHBNYXNrWzBdKTtcbiAgICAgIH1cbiAgICAgIGxldCB6ID0gSy5kb3QoaW5wdXRzLCB0aGlzLmtlcm5lbC5yZWFkKCkpO1xuICAgICAgaWYgKDAgPCB0aGlzLnJlY3VycmVudERyb3BvdXQgJiYgdGhpcy5yZWN1cnJlbnREcm9wb3V0IDwgMSkge1xuICAgICAgICBoVE1pbnVzMSA9IHRmYy5tdWwoaFRNaW51czEsIHJlY0RwTWFza1swXSk7XG4gICAgICB9XG4gICAgICB6ID0gdGZjLmFkZCh6LCBLLmRvdChoVE1pbnVzMSwgdGhpcy5yZWN1cnJlbnRLZXJuZWwucmVhZCgpKSk7XG4gICAgICBpZiAodGhpcy51c2VCaWFzKSB7XG4gICAgICAgIHogPSBLLmJpYXNBZGQoeiwgdGhpcy5iaWFzLnJlYWQoKSk7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IFt6MCwgejEsIHoyLCB6M10gPSB0ZmMuc3BsaXQoeiwgNCwgei5yYW5rIC0gMSk7XG5cbiAgICAgIGkgPSB0aGlzLnJlY3VycmVudEFjdGl2YXRpb24uYXBwbHkoejApO1xuICAgICAgZiA9IHRoaXMucmVjdXJyZW50QWN0aXZhdGlvbi5hcHBseSh6MSk7XG4gICAgICBjID0gdGZjLmFkZCh0ZmMubXVsKGYsIGNUTWludXMxKSwgdGZjLm11bChpLCB0aGlzLmFjdGl2YXRpb24uYXBwbHkoejIpKSk7XG4gICAgICBvID0gdGhpcy5yZWN1cnJlbnRBY3RpdmF0aW9uLmFwcGx5KHozKTtcblxuICAgICAgY29uc3QgaCA9IHRmYy5tdWwobywgdGhpcy5hY3RpdmF0aW9uLmFwcGx5KGMpKTtcbiAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCB1c2VfbGVhcm5pbmdfcGhhc2UgZmxhZyBwcm9wZXJseS5cbiAgICAgIHJldHVybiBbaCwgaCwgY107XG4gICAgfSk7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG5cbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtcbiAgICAgIHVuaXRzOiB0aGlzLnVuaXRzLFxuICAgICAgYWN0aXZhdGlvbjogc2VyaWFsaXplQWN0aXZhdGlvbih0aGlzLmFjdGl2YXRpb24pLFxuICAgICAgcmVjdXJyZW50QWN0aXZhdGlvbjogc2VyaWFsaXplQWN0aXZhdGlvbih0aGlzLnJlY3VycmVudEFjdGl2YXRpb24pLFxuICAgICAgdXNlQmlhczogdGhpcy51c2VCaWFzLFxuICAgICAga2VybmVsSW5pdGlhbGl6ZXI6IHNlcmlhbGl6ZUluaXRpYWxpemVyKHRoaXMua2VybmVsSW5pdGlhbGl6ZXIpLFxuICAgICAgcmVjdXJyZW50SW5pdGlhbGl6ZXI6IHNlcmlhbGl6ZUluaXRpYWxpemVyKHRoaXMucmVjdXJyZW50SW5pdGlhbGl6ZXIpLFxuICAgICAgYmlhc0luaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLmJpYXNJbml0aWFsaXplciksXG4gICAgICB1bml0Rm9yZ2V0QmlhczogdGhpcy51bml0Rm9yZ2V0QmlhcyxcbiAgICAgIGtlcm5lbFJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLmtlcm5lbFJlZ3VsYXJpemVyKSxcbiAgICAgIHJlY3VycmVudFJlZ3VsYXJpemVyOiBzZXJpYWxpemVSZWd1bGFyaXplcih0aGlzLnJlY3VycmVudFJlZ3VsYXJpemVyKSxcbiAgICAgIGJpYXNSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5iaWFzUmVndWxhcml6ZXIpLFxuICAgICAgYWN0aXZpdHlSZWd1bGFyaXplcjogc2VyaWFsaXplUmVndWxhcml6ZXIodGhpcy5hY3Rpdml0eVJlZ3VsYXJpemVyKSxcbiAgICAgIGtlcm5lbENvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5rZXJuZWxDb25zdHJhaW50KSxcbiAgICAgIHJlY3VycmVudENvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5yZWN1cnJlbnRDb25zdHJhaW50KSxcbiAgICAgIGJpYXNDb25zdHJhaW50OiBzZXJpYWxpemVDb25zdHJhaW50KHRoaXMuYmlhc0NvbnN0cmFpbnQpLFxuICAgICAgZHJvcG91dDogdGhpcy5kcm9wb3V0LFxuICAgICAgcmVjdXJyZW50RHJvcG91dDogdGhpcy5yZWN1cnJlbnREcm9wb3V0LFxuICAgICAgaW1wbGVtZW50YXRpb246IHRoaXMuaW1wbGVtZW50YXRpb24sXG4gICAgfTtcblxuICAgIHJldHVybiB7Li4uYmFzZUNvbmZpZywgLi4uY29uZmlnfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKExTVE1DZWxsKTtcblxuLy8gUG9ydGluZyBOb3RlOiBTaW5jZSB0aGlzIGlzIGEgc3VwZXJzZXQgb2YgU2ltcGxlUk5OTGF5ZXJDb25maWcsIHdlIGluaGVyaXRcbi8vICAgZnJvbSB0aGF0IGludGVyZmFjZSBpbnN0ZWFkIG9mIHJlcGVhdGluZyB0aGUgZmllbGRzIGhlcmUuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgTFNUTUxheWVyQXJncyBleHRlbmRzIFNpbXBsZVJOTkxheWVyQXJncyB7XG4gIC8qKlxuICAgKiBBY3RpdmF0aW9uIGZ1bmN0aW9uIHRvIHVzZSBmb3IgdGhlIHJlY3VycmVudCBzdGVwLlxuICAgKlxuICAgKiBEZWZhdWx0cyB0byBoYXJkIHNpZ21vaWQgKGBoYXJkU2lnbW9pZGApLlxuICAgKlxuICAgKiBJZiBgbnVsbGAsIG5vIGFjdGl2YXRpb24gaXMgYXBwbGllZC5cbiAgICovXG4gIHJlY3VycmVudEFjdGl2YXRpb24/OiBBY3RpdmF0aW9uSWRlbnRpZmllcjtcblxuICAvKipcbiAgICogSWYgYHRydWVgLCBhZGQgMSB0byB0aGUgYmlhcyBvZiB0aGUgZm9yZ2V0IGdhdGUgYXQgaW5pdGlhbGl6YXRpb24uXG4gICAqIFNldHRpbmcgaXQgdG8gYHRydWVgIHdpbGwgYWxzbyBmb3JjZSBgYmlhc0luaXRpYWxpemVyID0gJ3plcm9zJ2AuXG4gICAqIFRoaXMgaXMgcmVjb21tZW5kZWQgaW5cbiAgICogW0pvemVmb3dpY3ogZXRcbiAgICogYWwuXShodHRwOi8vd3d3LmptbHIub3JnL3Byb2NlZWRpbmdzL3BhcGVycy92Mzcvam96ZWZvd2ljejE1LnBkZilcbiAgICovXG4gIHVuaXRGb3JnZXRCaWFzPzogYm9vbGVhbjtcblxuICAvKipcbiAgICogSW1wbGVtZW50YXRpb24gbW9kZSwgZWl0aGVyIDEgb3IgMi5cbiAgICogICBNb2RlIDEgd2lsbCBzdHJ1Y3R1cmUgaXRzIG9wZXJhdGlvbnMgYXMgYSBsYXJnZXIgbnVtYmVyIG9mXG4gICAqICAgc21hbGxlciBkb3QgcHJvZHVjdHMgYW5kIGFkZGl0aW9ucywgd2hlcmVhcyBtb2RlIDIgd2lsbFxuICAgKiAgIGJhdGNoIHRoZW0gaW50byBmZXdlciwgbGFyZ2VyIG9wZXJhdGlvbnMuIFRoZXNlIG1vZGVzIHdpbGxcbiAgICogICBoYXZlIGRpZmZlcmVudCBwZXJmb3JtYW5jZSBwcm9maWxlcyBvbiBkaWZmZXJlbnQgaGFyZHdhcmUgYW5kXG4gICAqICAgZm9yIGRpZmZlcmVudCBhcHBsaWNhdGlvbnMuXG4gICAqXG4gICAqIE5vdGU6IEZvciBzdXBlcmlvciBwZXJmb3JtYW5jZSwgVGVuc29yRmxvdy5qcyBhbHdheXMgdXNlcyBpbXBsZW1lbnRhdGlvblxuICAgKiAyLCByZWdhcmRsZXNzIG9mIHRoZSBhY3R1YWwgdmFsdWUgb2YgdGhpcyBjb25maWcgZmllbGQuXG4gICAqL1xuICBpbXBsZW1lbnRhdGlvbj86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIExTVE0gZXh0ZW5kcyBSTk4ge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIG92ZXJyaWRlIGNsYXNzTmFtZSA9ICdMU1RNJztcbiAgY29uc3RydWN0b3IoYXJnczogTFNUTUxheWVyQXJncykge1xuICAgIGlmIChhcmdzLmltcGxlbWVudGF0aW9uID09PSAwKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgJ2BpbXBsZW1lbnRhdGlvbj0wYCBoYXMgYmVlbiBkZXByZWNhdGVkLCBhbmQgbm93IGRlZmF1bHRzIHRvICcgK1xuICAgICAgICAgICdgaW1wbGVtZW50YXRpb249MWAuIFBsZWFzZSB1cGRhdGUgeW91ciBsYXllciBjYWxsLicpO1xuICAgIH1cbiAgICBhcmdzLmNlbGwgPSBuZXcgTFNUTUNlbGwoYXJncyk7XG4gICAgc3VwZXIoYXJncyBhcyBSTk5MYXllckFyZ3MpO1xuICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBhY3Rpdml0eVJlZ3VsYXJpemVyLlxuICB9XG5cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGlmICh0aGlzLmNlbGwuZHJvcG91dE1hc2sgIT0gbnVsbCkge1xuICAgICAgICB0ZmMuZGlzcG9zZSh0aGlzLmNlbGwuZHJvcG91dE1hc2spO1xuICAgICAgICB0aGlzLmNlbGwuZHJvcG91dE1hc2sgPSBudWxsO1xuICAgICAgfVxuICAgICAgaWYgKHRoaXMuY2VsbC5yZWN1cnJlbnREcm9wb3V0TWFzayAhPSBudWxsKSB7XG4gICAgICAgIHRmYy5kaXNwb3NlKHRoaXMuY2VsbC5yZWN1cnJlbnREcm9wb3V0TWFzayk7XG4gICAgICAgIHRoaXMuY2VsbC5yZWN1cnJlbnREcm9wb3V0TWFzayA9IG51bGw7XG4gICAgICB9XG4gICAgICBjb25zdCBtYXNrID0ga3dhcmdzID09IG51bGwgPyBudWxsIDoga3dhcmdzWydtYXNrJ107XG4gICAgICBjb25zdCB0cmFpbmluZyA9IGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1sndHJhaW5pbmcnXTtcbiAgICAgIGNvbnN0IGluaXRpYWxTdGF0ZTogVGVuc29yW10gPVxuICAgICAgICAgIGt3YXJncyA9PSBudWxsID8gbnVsbCA6IGt3YXJnc1snaW5pdGlhbFN0YXRlJ107XG4gICAgICByZXR1cm4gc3VwZXIuY2FsbChpbnB1dHMsIHttYXNrLCB0cmFpbmluZywgaW5pdGlhbFN0YXRlfSk7XG4gICAgfSk7XG4gIH1cblxuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIG92ZXJyaWRlIGZyb21Db25maWc8VCBleHRlbmRzIHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlPihcbiAgICAgIGNsczogc2VyaWFsaXphdGlvbi5TZXJpYWxpemFibGVDb25zdHJ1Y3RvcjxUPixcbiAgICAgIGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0KTogVCB7XG4gICAgaWYgKGNvbmZpZ1snaW1wbG1lbnRhdGlvbiddID09PSAwKSB7XG4gICAgICBjb25maWdbJ2ltcGxlbWVudGF0aW9uJ10gPSAxO1xuICAgIH1cbiAgICByZXR1cm4gbmV3IGNscyhjb25maWcpO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoTFNUTSk7XG5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBTdGFja2VkUk5OQ2VsbHNBcmdzIGV4dGVuZHMgTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIEFuIGBBcnJheWAgb2YgYFJOTkNlbGxgIGluc3RhbmNlcy5cbiAgICovXG4gIGNlbGxzOiBSTk5DZWxsW107XG59XG5cbmV4cG9ydCBjbGFzcyBTdGFja2VkUk5OQ2VsbHMgZXh0ZW5kcyBSTk5DZWxsIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnU3RhY2tlZFJOTkNlbGxzJztcbiAgcHJvdGVjdGVkIGNlbGxzOiBSTk5DZWxsW107XG5cbiAgY29uc3RydWN0b3IoYXJnczogU3RhY2tlZFJOTkNlbGxzQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMuY2VsbHMgPSBhcmdzLmNlbGxzO1xuICB9XG5cbiAgZ2V0IHN0YXRlU2l6ZSgpOiBudW1iZXJbXSB7XG4gICAgLy8gU3RhdGVzIGFyZSBhIGZsYXQgbGlzdCBpbiByZXZlcnNlIG9yZGVyIG9mIHRoZSBjZWxsIHN0YWNrLlxuICAgIC8vIFRoaXMgYWxsb3dzIHBlcnNlcnZpbmcgdGhlIHJlcXVpcmVtZW50IGBzdGFjay5zdGF0ZXNpemVbMF0gPT09XG4gICAgLy8gb3V0cHV0RGltYC4gRS5nLiwgc3RhdGVzIG9mIGEgMi1sYXllciBMU1RNIHdvdWxkIGJlIGBbaDIsIGMyLCBoMSwgYzFdYCxcbiAgICAvLyBhc3N1bWluZyBvbmUgTFNUTSBoYXMgc3RhdGVzIGBbaCwgY11gLlxuICAgIGNvbnN0IHN0YXRlU2l6ZTogbnVtYmVyW10gPSBbXTtcbiAgICBmb3IgKGNvbnN0IGNlbGwgb2YgdGhpcy5jZWxscy5zbGljZSgpLnJldmVyc2UoKSkge1xuICAgICAgaWYgKEFycmF5LmlzQXJyYXkoY2VsbC5zdGF0ZVNpemUpKSB7XG4gICAgICAgIHN0YXRlU2l6ZS5wdXNoKC4uLmNlbGwuc3RhdGVTaXplKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHN0YXRlU2l6ZS5wdXNoKGNlbGwuc3RhdGVTaXplKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHN0YXRlU2l6ZTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpbnB1dHMgPSBpbnB1dHMgYXMgVGVuc29yW107XG4gICAgICBsZXQgc3RhdGVzID0gaW5wdXRzLnNsaWNlKDEpO1xuXG4gICAgICAvLyBSZWNvdmVyIHBlci1jZWxsIHN0YXRlcy5cbiAgICAgIGNvbnN0IG5lc3RlZFN0YXRlczogVGVuc29yW11bXSA9IFtdO1xuICAgICAgZm9yIChjb25zdCBjZWxsIG9mIHRoaXMuY2VsbHMuc2xpY2UoKS5yZXZlcnNlKCkpIHtcbiAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkoY2VsbC5zdGF0ZVNpemUpKSB7XG4gICAgICAgICAgbmVzdGVkU3RhdGVzLnB1c2goc3RhdGVzLnNwbGljZSgwLCBjZWxsLnN0YXRlU2l6ZS5sZW5ndGgpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBuZXN0ZWRTdGF0ZXMucHVzaChzdGF0ZXMuc3BsaWNlKDAsIDEpKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgbmVzdGVkU3RhdGVzLnJldmVyc2UoKTtcblxuICAgICAgLy8gQ2FsbCB0aGUgY2VsbHMgaW4gb3JkZXIgYW5kIHN0b3JlIHRoZSByZXR1cm5lZCBzdGF0ZXMuXG4gICAgICBjb25zdCBuZXdOZXN0ZWRTdGF0ZXM6IFRlbnNvcltdW10gPSBbXTtcbiAgICAgIGxldCBjYWxsSW5wdXRzOiBUZW5zb3JbXTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5jZWxscy5sZW5ndGg7ICsraSkge1xuICAgICAgICBjb25zdCBjZWxsID0gdGhpcy5jZWxsc1tpXTtcbiAgICAgICAgc3RhdGVzID0gbmVzdGVkU3RhdGVzW2ldO1xuICAgICAgICAvLyBUT0RPKGNhaXMpOiBUYWtlIGNhcmUgb2YgY29uc3RhbnRzLlxuICAgICAgICBpZiAoaSA9PT0gMCkge1xuICAgICAgICAgIGNhbGxJbnB1dHMgPSBbaW5wdXRzWzBdXS5jb25jYXQoc3RhdGVzKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBjYWxsSW5wdXRzID0gW2NhbGxJbnB1dHNbMF1dLmNvbmNhdChzdGF0ZXMpO1xuICAgICAgICB9XG4gICAgICAgIGNhbGxJbnB1dHMgPSBjZWxsLmNhbGwoY2FsbElucHV0cywga3dhcmdzKSBhcyBUZW5zb3JbXTtcbiAgICAgICAgbmV3TmVzdGVkU3RhdGVzLnB1c2goY2FsbElucHV0cy5zbGljZSgxKSk7XG4gICAgICB9XG5cbiAgICAgIC8vIEZvcm1hdCB0aGUgbmV3IHN0YXRlcyBhcyBhIGZsYXQgbGlzdCBpbiByZXZlcnNlIGNlbGwgb3JkZXIuXG4gICAgICBzdGF0ZXMgPSBbXTtcbiAgICAgIGZvciAoY29uc3QgY2VsbFN0YXRlcyBvZiBuZXdOZXN0ZWRTdGF0ZXMuc2xpY2UoKS5yZXZlcnNlKCkpIHtcbiAgICAgICAgc3RhdGVzLnB1c2goLi4uY2VsbFN0YXRlcyk7XG4gICAgICB9XG4gICAgICByZXR1cm4gW2NhbGxJbnB1dHNbMF1dLmNvbmNhdChzdGF0ZXMpO1xuICAgIH0pO1xuICB9XG5cbiAgcHVibGljIG92ZXJyaWRlIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICBpZiAoaXNBcnJheU9mU2hhcGVzKGlucHV0U2hhcGUpKSB7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBUYWtlIGNhcmUgb2YgaW5wdXQgY29uc3RhbnRzLlxuICAgICAgLy8gY29uc3QgY29uc3RhbnRTaGFwZSA9IGlucHV0U2hhcGUuc2xpY2UoMSk7XG4gICAgICBpbnB1dFNoYXBlID0gKGlucHV0U2hhcGUgYXMgU2hhcGVbXSlbMF07XG4gICAgfVxuICAgIGlucHV0U2hhcGUgPSBpbnB1dFNoYXBlIGFzIFNoYXBlO1xuICAgIGxldCBvdXRwdXREaW06IG51bWJlcjtcbiAgICB0aGlzLmNlbGxzLmZvckVhY2goKGNlbGwsIGkpID0+IHtcbiAgICAgIG5hbWVTY29wZShgUk5OQ2VsbF8ke2l9YCwgKCkgPT4ge1xuICAgICAgICAvLyBUT0RPKGNhaXMpOiBUYWtlIGNhcmUgb2YgaW5wdXQgY29uc3RhbnRzLlxuXG4gICAgICAgIGNlbGwuYnVpbGQoaW5wdXRTaGFwZSk7XG4gICAgICAgIGlmIChBcnJheS5pc0FycmF5KGNlbGwuc3RhdGVTaXplKSkge1xuICAgICAgICAgIG91dHB1dERpbSA9IGNlbGwuc3RhdGVTaXplWzBdO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIG91dHB1dERpbSA9IGNlbGwuc3RhdGVTaXplO1xuICAgICAgICB9XG4gICAgICAgIGlucHV0U2hhcGUgPSBbaW5wdXRTaGFwZVswXSwgb3V0cHV0RGltXSBhcyBTaGFwZTtcbiAgICAgIH0pO1xuICAgIH0pO1xuICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuXG4gICAgY29uc3QgZ2V0Q2VsbENvbmZpZyA9IChjZWxsOiBSTk5DZWxsKSA9PiB7XG4gICAgICByZXR1cm4ge1xuICAgICAgICAnY2xhc3NOYW1lJzogY2VsbC5nZXRDbGFzc05hbWUoKSxcbiAgICAgICAgJ2NvbmZpZyc6IGNlbGwuZ2V0Q29uZmlnKCksXG4gICAgICB9O1xuICAgIH07XG5cbiAgICBjb25zdCBjZWxsQ29uZmlncyA9IHRoaXMuY2VsbHMubWFwKGdldENlbGxDb25maWcpO1xuXG4gICAgY29uc3QgY29uZmlnID0geydjZWxscyc6IGNlbGxDb25maWdzfTtcblxuICAgIHJldHVybiB7Li4uYmFzZUNvbmZpZywgLi4uY29uZmlnfTtcbiAgfVxuXG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgb3ZlcnJpZGUgZnJvbUNvbmZpZzxUIGV4dGVuZHMgc2VyaWFsaXphdGlvbi5TZXJpYWxpemFibGU+KFxuICAgICAgY2xzOiBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZUNvbnN0cnVjdG9yPFQ+LFxuICAgICAgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QsXG4gICAgICBjdXN0b21PYmplY3RzID0ge30gYXMgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0KTogVCB7XG4gICAgY29uc3QgY2VsbHM6IFJOTkNlbGxbXSA9IFtdO1xuICAgIGZvciAoY29uc3QgY2VsbENvbmZpZyBvZiAoY29uZmlnWydjZWxscyddIGFzIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdFtdKSkge1xuICAgICAgY2VsbHMucHVzaChkZXNlcmlhbGl6ZShjZWxsQ29uZmlnLCBjdXN0b21PYmplY3RzKSBhcyBSTk5DZWxsKTtcbiAgICB9XG4gICAgcmV0dXJuIG5ldyBjbHMoe2NlbGxzfSk7XG4gIH1cblxuICBvdmVycmlkZSBnZXQgdHJhaW5hYmxlV2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIGlmICghdGhpcy50cmFpbmFibGUpIHtcbiAgICAgIHJldHVybiBbXTtcbiAgICB9XG4gICAgY29uc3Qgd2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdID0gW107XG4gICAgZm9yIChjb25zdCBjZWxsIG9mIHRoaXMuY2VsbHMpIHtcbiAgICAgIHdlaWdodHMucHVzaCguLi5jZWxsLnRyYWluYWJsZVdlaWdodHMpO1xuICAgIH1cbiAgICByZXR1cm4gd2VpZ2h0cztcbiAgfVxuXG4gIG92ZXJyaWRlIGdldCBub25UcmFpbmFibGVXZWlnaHRzKCk6IExheWVyVmFyaWFibGVbXSB7XG4gICAgY29uc3Qgd2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdID0gW107XG4gICAgZm9yIChjb25zdCBjZWxsIG9mIHRoaXMuY2VsbHMpIHtcbiAgICAgIHdlaWdodHMucHVzaCguLi5jZWxsLm5vblRyYWluYWJsZVdlaWdodHMpO1xuICAgIH1cbiAgICBpZiAoIXRoaXMudHJhaW5hYmxlKSB7XG4gICAgICBjb25zdCB0cmFpbmFibGVXZWlnaHRzOiBMYXllclZhcmlhYmxlW10gPSBbXTtcbiAgICAgIGZvciAoY29uc3QgY2VsbCBvZiB0aGlzLmNlbGxzKSB7XG4gICAgICAgIHRyYWluYWJsZVdlaWdodHMucHVzaCguLi5jZWxsLnRyYWluYWJsZVdlaWdodHMpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHRyYWluYWJsZVdlaWdodHMuY29uY2F0KHdlaWdodHMpO1xuICAgIH1cbiAgICByZXR1cm4gd2VpZ2h0cztcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXRyaWV2ZSB0aGUgd2VpZ2h0cyBvZiBhIHRoZSBtb2RlbC5cbiAgICpcbiAgICogQHJldHVybnMgQSBmbGF0IGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzLlxuICAgKi9cbiAgb3ZlcnJpZGUgZ2V0V2VpZ2h0cygpOiBUZW5zb3JbXSB7XG4gICAgY29uc3Qgd2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdID0gW107XG4gICAgZm9yIChjb25zdCBjZWxsIG9mIHRoaXMuY2VsbHMpIHtcbiAgICAgIHdlaWdodHMucHVzaCguLi5jZWxsLndlaWdodHMpO1xuICAgIH1cbiAgICByZXR1cm4gYmF0Y2hHZXRWYWx1ZSh3ZWlnaHRzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTZXQgdGhlIHdlaWdodHMgb2YgdGhlIG1vZGVsLlxuICAgKlxuICAgKiBAcGFyYW0gd2VpZ2h0cyBBbiBgQXJyYXlgIG9mIGB0Zi5UZW5zb3JgcyB3aXRoIHNoYXBlcyBhbmQgdHlwZXMgbWF0Y2hpbmdcbiAgICogICAgIHRoZSBvdXRwdXQgb2YgYGdldFdlaWdodHMoKWAuXG4gICAqL1xuICBvdmVycmlkZSBzZXRXZWlnaHRzKHdlaWdodHM6IFRlbnNvcltdKTogdm9pZCB7XG4gICAgY29uc3QgdHVwbGVzOiBBcnJheTxbTGF5ZXJWYXJpYWJsZSwgVGVuc29yXT4gPSBbXTtcbiAgICBmb3IgKGNvbnN0IGNlbGwgb2YgdGhpcy5jZWxscykge1xuICAgICAgY29uc3QgbnVtUGFyYW1zID0gY2VsbC53ZWlnaHRzLmxlbmd0aDtcbiAgICAgIGNvbnN0IGlucHV0V2VpZ2h0cyA9IHdlaWdodHMuc3BsaWNlKG51bVBhcmFtcyk7XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IGNlbGwud2VpZ2h0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICB0dXBsZXMucHVzaChbY2VsbC53ZWlnaHRzW2ldLCBpbnB1dFdlaWdodHNbaV1dKTtcbiAgICAgIH1cbiAgICB9XG4gICAgYmF0Y2hTZXRWYWx1ZSh0dXBsZXMpO1xuICB9XG5cbiAgLy8gVE9ETyhjYWlzKTogTWF5YmUgaW1wbGVtbnQgYGxvc3Nlc2AgYW5kIGBnZXRMb3NzZXNGb3JgLlxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFN0YWNrZWRSTk5DZWxscyk7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZW5lcmF0ZURyb3BvdXRNYXNrKGFyZ3M6IHtcbiAgb25lczogKCkgPT4gdGZjLlRlbnNvcixcbiAgcmF0ZTogbnVtYmVyLFxuICB0cmFpbmluZz86IGJvb2xlYW4sXG4gIGNvdW50PzogbnVtYmVyLFxuICBkcm9wb3V0RnVuYz86IEZ1bmN0aW9uLFxufSk6IHRmYy5UZW5zb3J8dGZjLlRlbnNvcltdIHtcbiAgY29uc3Qge29uZXMsIHJhdGUsIHRyYWluaW5nID0gZmFsc2UsIGNvdW50ID0gMSwgZHJvcG91dEZ1bmN9ID0gYXJncztcblxuICBjb25zdCBkcm9wcGVkSW5wdXRzID0gKCkgPT5cbiAgICAgIGRyb3BvdXRGdW5jICE9IG51bGwgPyBkcm9wb3V0RnVuYyhvbmVzKCksIHJhdGUpIDogSy5kcm9wb3V0KG9uZXMoKSwgcmF0ZSk7XG5cbiAgY29uc3QgY3JlYXRlTWFzayA9ICgpID0+IEsuaW5UcmFpblBoYXNlKGRyb3BwZWRJbnB1dHMsIG9uZXMsIHRyYWluaW5nKTtcblxuICAvLyBqdXN0IGluIGNhc2UgY291bnQgaXMgcHJvdmlkZWQgd2l0aCBudWxsIG9yIHVuZGVmaW5lZFxuICBpZiAoIWNvdW50IHx8IGNvdW50IDw9IDEpIHtcbiAgICByZXR1cm4gdGZjLmtlZXAoY3JlYXRlTWFzaygpLmNsb25lKCkpO1xuICB9XG5cbiAgY29uc3QgbWFza3MgPSBBcnJheShjb3VudCkuZmlsbCh1bmRlZmluZWQpLm1hcChjcmVhdGVNYXNrKTtcblxuICByZXR1cm4gbWFza3MubWFwKG0gPT4gdGZjLmtlZXAobS5jbG9uZSgpKSk7XG59XG4iXX0=