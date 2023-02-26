/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* Original Source: engine/training.py */
import * as tfc from '@tensorflow/tfjs-core';
import { io, Optimizer, scalar, serialization, Tensor, tensor1d, util } from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import { configureCallbacks, standardizeCallbacks } from '../base_callbacks';
import { nameScope } from '../common';
import { NotImplementedError, RuntimeError, ValueError } from '../errors';
import { deserialize } from '../layers/serialization';
import { disposeTensorsInLogs } from '../logs';
import * as losses from '../losses';
import * as Metrics from '../metrics';
import * as optimizers from '../optimizers';
import { checkUserDefinedMetadata } from '../user_defined_metadata';
import { count, pyListRepeat, singletonOrArray, toCamelCase, toSnakeCase, unique } from '../utils/generic_utils';
import { printSummary } from '../utils/layer_utils';
import { range } from '../utils/math_utils';
import { convertPythonicToTs } from '../utils/serialization_utils';
import { version } from '../version';
import { Container } from './container';
import { execute, FeedDict } from './executor';
import { evaluateDataset, fitDataset } from './training_dataset';
import { checkBatchSize, disposeNewTensors, ensureTensorsRank2OrHigher, makeBatches, sliceArrays, sliceArraysByIndices } from './training_tensors';
import { computeWeightedLoss, standardizeClassWeights, standardizeWeights } from './training_utils';
/**
 * Helper function for polymorphic input data: 1. singleton Tensor.
 */
export function isDataTensor(x) {
    return x instanceof Tensor;
}
/**
 * Helper function for polymorphic input data: 2. Array of Tensor.
 */
export function isDataArray(x) {
    return Array.isArray(x);
}
/**
 * Helper function for polymorphic input data: 3. "dict" of Tensor.
 */
export function isDataDict(x) {
    return !isDataTensor(x) && !isDataArray(x);
}
/**
 * Normalizes inputs and targets provided by users.
 * @param data User-provided input data (polymorphic).
 * @param names An Array of expected Tensor names.
 * @param shapes Optional Array of expected Tensor shapes.
 * @param checkBatchAxis Whether to check that the batch axis of the arrays
 *   match  the expected value found in `shapes`.
 * @param exceptionPrefix String prefix used for exception formatting.
 * @returns List of standardized input Tensors (one Tensor per model input).
 * @throws ValueError: in case of improperly formatted user data.
 */
export function standardizeInputData(data, names, shapes, checkBatchAxis = true, exceptionPrefix = '') {
    if (names == null || names.length === 0) {
        // Check for the case where the model expected no data, but some data got
        // sent.
        if (data != null) {
            let gotUnexpectedData = false;
            if (isDataArray(data) && data.length > 0) {
                gotUnexpectedData = true;
            }
            else if (isDataDict(data)) {
                for (const key in data) {
                    if (data.hasOwnProperty(key)) {
                        gotUnexpectedData = true;
                        break;
                    }
                }
            }
            else {
                // `data` is a singleton Tensor in this case.
                gotUnexpectedData = true;
            }
            if (gotUnexpectedData) {
                throw new ValueError(`Error when checking model ${exceptionPrefix} expected no data, ` +
                    `but got ${data}`);
            }
        }
        return [];
    }
    if (data == null) {
        return names.map(name => null);
    }
    let arrays;
    if (isDataDict(data)) {
        data = data;
        arrays = [];
        for (const name of names) {
            if (data[name] == null) {
                throw new ValueError(`No data provided for "${name}". Need data for each key in: ` +
                    `${names}`);
            }
            arrays.push(data[name]);
        }
    }
    else if (isDataArray(data)) {
        data = data;
        if (data.length !== names.length) {
            throw new ValueError(`Error when checking model ${exceptionPrefix}: the Array of ` +
                `Tensors that you are passing to your model is not the size the ` +
                `model expected. Expected to see ${names.length} Tensor(s), but ` +
                `instead got the following list of Tensor(s): ${data}`);
        }
        arrays = data;
    }
    else {
        data = data;
        if (names.length > 1) {
            throw new ValueError(`The model ${exceptionPrefix} expects ${names.length} Tensor(s), ` +
                `but only received one Tensor. Found: Tensor with shape ${data.shape}`);
        }
        arrays = [data];
    }
    arrays = ensureTensorsRank2OrHigher(arrays);
    // Check shape compatibility.
    if (shapes != null) {
        for (let i = 0; i < names.length; ++i) {
            if (shapes[i] == null) {
                continue;
            }
            const array = arrays[i];
            if (array.shape.length !== shapes[i].length) {
                throw new ValueError(`Error when checking ${exceptionPrefix}: expected ${names[i]} ` +
                    `to have ${shapes[i].length} dimension(s). but got array with ` +
                    `shape ${array.shape}`);
            }
            for (let j = 0; j < shapes[i].length; ++j) {
                if (j === 0 && !checkBatchAxis) {
                    // Skip the first (batch) axis.
                    continue;
                }
                const dim = array.shape[j];
                const refDim = shapes[i][j];
                if (refDim != null && refDim >= 0 && dim !== refDim) {
                    throw new ValueError(`${exceptionPrefix} expected a batch of elements where each ` +
                        `example has shape [${shapes[i].slice(1, shapes[i].length)}] ` +
                        `(i.e.,tensor shape [*,${shapes[i].slice(1, shapes[i].length)}])` +
                        ` but the ${exceptionPrefix} received an input with ${array.shape[0]}` +
                        ` examples, each with shape [${array.shape.slice(1, array.shape.length)}]` +
                        ` (tensor shape [${array.shape}])`);
                }
            }
        }
    }
    return arrays;
}
/**
 * User input validation for Tensors.
 * @param inputs `Array` of `tf.Tensor`s for inputs.
 * @param targets `Array` of `tf.Tensor`s for targets.
 * @param weights Optional `Array` of `tf.Tensor`s for sample weights.
 * @throws ValueError: in case of incorrectly formatted data.
 */
export function checkArrayLengths(inputs, targets, weights) {
    const setX = unique(inputs.map(input => input.shape[0]));
    setX.sort();
    const setY = unique(targets.map(target => target.shape[0]));
    setY.sort();
    // TODO(cais): Check `weights` as well.
    if (setX.length > 1) {
        throw new ValueError(`All input Tensors (x) should have the same number of samples. ` +
            `Got array shapes: ` +
            `${JSON.stringify(inputs.map(input => input.shape))}`);
    }
    if (setY.length > 1) {
        throw new ValueError(`All target Tensors (y) should have the same number of samples. ` +
            `Got array shapes: ` +
            `${JSON.stringify(targets.map(target => target.shape))}`);
    }
    if (setX.length > 0 && setY.length > 0 && !util.arraysEqual(setX, setY)) {
        throw new ValueError(`Input Tensors should have the same number of samples as target ` +
            `Tensors. Found ${setX[0]} input sample(s) and ${setY[0]} target ` +
            `sample(s).`);
    }
}
/**
 * Validation on the compatibility of targes and loss functions.
 *
 * This helps prevent users from using loss functions incorrectly.
 *
 * @param targets `Array` of `tf.Tensor`s of targets.
 * @param lossFns `Array` of loss functions.
 * @param outputShapes `Array` of shapes of model outputs.
 */
function checkLossAndTargetCompatibility(targets, lossFns, outputShapes) {
    // TODO(cais): Dedicated test coverage?
    const keyLosses = [
        losses.meanSquaredError, losses.binaryCrossentropy,
        losses.categoricalCrossentropy
    ];
    for (let i = 0; i < targets.length; ++i) {
        const y = targets[i];
        const loss = lossFns[i];
        const shape = outputShapes[i];
        if (loss == null) {
            continue;
        }
        if (loss === losses.categoricalCrossentropy) {
            if (y.shape[y.shape.length - 1] === 1) {
                throw new ValueError(`You are passing a target array of shape ${y.shape} while using ` +
                    `a loss 'categorical_crossentropy'. 'categorical_crossentropy'` +
                    `expects targets to be binary matrices (1s and 0s) of shape ` +
                    `[samples, classes].`);
                // TODO(cais): Example code in error message.
            }
        }
        if (keyLosses.indexOf(loss) !== -1) {
            const slicedYShape = y.shape.slice(1);
            const slicedShape = shape.slice(1);
            for (let j = 0; j < slicedYShape.length; ++j) {
                const targetDim = slicedYShape[j];
                const outDim = slicedShape[j];
                if (outDim != null && targetDim !== outDim) {
                    throw new ValueError(`A target Tensor with shape ${y.shape} was passed for an ` +
                        `output of shape ${shape}, while using a loss function that ` +
                        `expects targets to have the same shape as the output.`);
                }
            }
        }
    }
}
/**
 * Check inputs provided by the user.
 *
 * Porting Note: This corresponds to _standardize_input_data() in Python
 *   Keras. Because of the strong typing in TF.js, we do not need to convert
 *   the data. Specifically:
 *   1) in PyKeras, `data` can be `DataFrame` instances from pandas, for
 *      example. We don't need to worry about that here because there is no
 *      widely popular javascript/typesdcript equivalent of pandas (so far).
 *      If one becomes available in the future, we can add support.
 *   2) in PyKeras, inputs can be Python dict. But here we are stipulating
 * that the data is either a single `tf.Tensor` or an Array of `tf.Tensor`s. We
 * may add support for `Object` data inputs in the future when the need
 * arises.
 *
 * Instead, we perform basic checks for number of parameters and shapes.
 *
 * @param data: The input data.
 * @param names: Name for the inputs, from the model.
 * @param shapes: Expected shapes for the input data, from the model.
 * @param checkBatchAxis: Whether the size along the batch axis (i.e., the
 *   first dimension) will be checked for matching.
 * @param exceptionPrefix: Execption prefix message, used in generating error
 *   messages.
 * @throws ValueError: on incorrect number of inputs or mismatches in shapes.
 */
function checkInputData(data, names, shapes, checkBatchAxis = true, exceptionPrefix = '') {
    let arrays;
    if (Array.isArray(data)) {
        if (data.length !== names.length) {
            throw new ValueError(`Error when checking model ${exceptionPrefix}: the Array of ` +
                `Tensors that you are passing to your model is not the size the ` +
                `the model expected. Expected to see ${names.length} Tensor(s),` +
                ` but instead got ${data.length} Tensors(s).`);
        }
        arrays = data;
    }
    else {
        if (names.length > 1) {
            throw new ValueError(`The model expects ${names.length} ${exceptionPrefix} Tensors, ` +
                `but only received one Tensor. Found: array with shape ` +
                `${JSON.stringify(data.shape)}.`);
        }
        arrays = [data];
    }
    if (shapes != null) {
        for (let i = 0; i < names.length; ++i) {
            if (shapes[i] == null) {
                continue;
            }
            const array = arrays[i];
            if (array.shape.length !== shapes[i].length) {
                throw new ValueError(`Error when checking ${exceptionPrefix}: expected ${names[i]} ` +
                    `to have ${shapes[i].length} dimension(s), but got array with ` +
                    `shape ${JSON.stringify(array.shape)}`);
            }
            for (let j = 0; j < shapes[i].length; ++j) {
                if (j === 0 && !checkBatchAxis) {
                    continue;
                }
                const dim = array.shape[j];
                const refDim = shapes[i][j];
                if (refDim != null) {
                    if (refDim !== dim) {
                        throw new ValueError(`Error when checking ${exceptionPrefix}: expected ` +
                            `${names[i]} to have shape ${JSON.stringify(shapes[i])} but ` +
                            `got array with shape ${JSON.stringify(array.shape)}.`);
                    }
                }
            }
        }
    }
}
/**
 * Maps metric functions to model outputs.
 * @param metrics An shortcut strings name, metric function, `Array` or dict
 *   (`Object`) of metric functions.
 * @param outputNames An `Array` of the names of model outputs.
 * @returns An `Array` (one entry per model output) of `Array` of metric
 *   functions. For instance, if the model has 2 outputs, and for the first
 *   output we want to compute `binaryAccuracy` and `binaryCrossentropy`,
 *   and just `binaryAccuracy` for the second output, the `Array` would look
 *   like:
 *     `[[binaryAccuracy, binaryCrossentropy],  [binaryAccuracy]]`
 * @throws TypeError: incompatible metrics format.
 */
export function collectMetrics(metrics, outputNames) {
    if (metrics == null || Array.isArray(metrics) && metrics.length === 0) {
        return outputNames.map(name => []);
    }
    let wrappedMetrics;
    if (typeof metrics === 'string' || typeof metrics === 'function') {
        wrappedMetrics = [metrics];
    }
    else if (Array.isArray(metrics) || typeof metrics === 'object') {
        wrappedMetrics = metrics;
    }
    else {
        throw new TypeError('Type of metrics argument not understood. Expected an string,' +
            `function, Array, or Object, found: ${metrics}`);
    }
    if (Array.isArray(wrappedMetrics)) {
        // We then apply all metrics to all outputs.
        return outputNames.map(name => wrappedMetrics);
    }
    else {
        // In this case, metrics is a dict.
        const nestedMetrics = [];
        for (const name of outputNames) {
            let outputMetrics = wrappedMetrics.hasOwnProperty(name) ? wrappedMetrics[name] : [];
            if (!Array.isArray(outputMetrics)) {
                outputMetrics = [outputMetrics];
            }
            nestedMetrics.push(outputMetrics);
        }
        return nestedMetrics;
    }
}
const LAYERS_MODEL_FORMAT_NAME = 'layers-model';
/**
 * A `tf.LayersModel` is a directed, acyclic graph of `tf.Layer`s plus methods
 * for training, evaluation, prediction and saving.
 *
 * `tf.LayersModel` is the basic unit of training, inference and evaluation in
 * TensorFlow.js. To create a `tf.LayersModel`, use `tf.LayersModel`.
 *
 * See also:
 *   `tf.Sequential`, `tf.loadLayersModel`.
 *
 * @doc {heading: 'Models', subheading: 'Classes'}
 */
export class LayersModel extends Container {
    constructor(args) {
        super(args);
        this.isTraining = false;
    }
    /**
     * Print a text summary of the model's layers.
     *
     * The summary includes
     * - Name and type of all layers that comprise the model.
     * - Output shape(s) of the layers
     * - Number of weight parameters of each layer
     * - If the model has non-sequential-like topology, the inputs each layer
     *   receives
     * - The total number of trainable and non-trainable parameters of the model.
     *
     * ```js
     * const input1 = tf.input({shape: [10]});
     * const input2 = tf.input({shape: [20]});
     * const dense1 = tf.layers.dense({units: 4}).apply(input1);
     * const dense2 = tf.layers.dense({units: 8}).apply(input2);
     * const concat = tf.layers.concatenate().apply([dense1, dense2]);
     * const output =
     *     tf.layers.dense({units: 3, activation: 'softmax'}).apply(concat);
     *
     * const model = tf.model({inputs: [input1, input2], outputs: output});
     * model.summary();
     * ```
     *
     * @param lineLength Custom line length, in number of characters.
     * @param positions Custom widths of each of the columns, as either
     *   fractions of `lineLength` (e.g., `[0.5, 0.75, 1]`) or absolute number
     *   of characters (e.g., `[30, 50, 65]`). Each number corresponds to
     *   right-most (i.e., ending) position of a column.
     * @param printFn Custom print function. Can be used to replace the default
     *   `console.log`. For example, you can use `x => {}` to mute the printed
     *   messages in the console.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    summary(lineLength, positions, printFn = console.log) {
        if (!this.built) {
            throw new ValueError(`This model has never been called, thus its weights have not been ` +
                `created yet. So no summary can be displayed. Build the model ` +
                `first (e.g., by calling it on some test data).`);
        }
        printSummary(this, lineLength, positions, printFn);
    }
    /**
     * Configures and prepares the model for training and evaluation.  Compiling
     * outfits the model with an optimizer, loss, and/or metrics.  Calling `fit`
     * or `evaluate` on an un-compiled model will throw an error.
     *
     * @param args a `ModelCompileArgs` specifying the loss, optimizer, and
     * metrics to be used for fitting and evaluating this model.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    compile(args) {
        if (args.loss == null) {
            args.loss = [];
        }
        this.loss = args.loss;
        if (typeof args.optimizer === 'string') {
            this.optimizer_ = optimizers.getOptimizer(args.optimizer);
            this.isOptimizerOwned = true;
        }
        else {
            if (!(args.optimizer instanceof Optimizer)) {
                throw new ValueError(`User-defined optimizer must be an instance of tf.Optimizer.`);
            }
            this.optimizer_ = args.optimizer;
            this.isOptimizerOwned = false;
        }
        // TODO(cais): Add lossWeights.
        // TODO(cais): Add sampleWeightMode.
        // Prepare loss functions.
        let lossFunctions = [];
        if (!Array.isArray(args.loss) && typeof args.loss !== 'string' &&
            typeof args.loss !== 'function') {
            args.loss = args.loss;
            for (const name in args.loss) {
                if (this.outputNames.indexOf(name) === -1) {
                    throw new ValueError(`Unknown entry in loss dictionary: "${name}". ` +
                        `Only expected the following keys: ${this.outputNames}`);
                }
            }
            for (const name of this.outputNames) {
                if (args.loss[name] == null) {
                    console.warn(`Output "${name}" is missing from loss dictionary. We assume ` +
                        `this was done on purpose, and we will not be expecting data ` +
                        `to be passed to ${name} during training`);
                }
                lossFunctions.push(losses.get(args.loss[name]));
            }
        }
        else if (Array.isArray(args.loss)) {
            if (args.loss.length !== this.outputs.length) {
                throw new ValueError(`When passing an Array as loss, it should have one entry per ` +
                    `model output. The model has ${this.outputs.length} output(s), ` +
                    `but you passed loss=${args.loss}.`);
            }
            const theLosses = args.loss;
            lossFunctions = theLosses.map(l => losses.get(l));
        }
        else {
            const lossFunction = losses.get(args.loss);
            this.outputs.forEach(_ => {
                lossFunctions.push(lossFunction);
            });
        }
        this.lossFunctions = lossFunctions;
        this.feedOutputNames = [];
        this.feedOutputShapes = [];
        this.feedLossFns = [];
        for (let i = 0; i < this.outputs.length; ++i) {
            // TODO(cais): Logic for skipping target(s).
            const shape = this.internalOutputShapes[i];
            const name = this.outputNames[i];
            this.feedOutputNames.push(name);
            this.feedOutputShapes.push(shape);
            this.feedLossFns.push(this.lossFunctions[i]);
        }
        // TODO(cais): Add logic for output masks.
        // TODO(cais): Add logic for sample weights.
        const skipTargetIndices = [];
        // Prepare metrics.
        this.metrics = args.metrics;
        // TODO(cais): Add weightedMetrics.
        this.metricsNames = ['loss'];
        this.metricsTensors = [];
        // Compute total loss.
        // Porting Note: In PyKeras, metrics_tensors are symbolic tensor objects.
        //   Here, metricsTensors are TypeScript functions. This difference is due
        //   to the difference in symbolic/imperative property of the backends.
        nameScope('loss', () => {
            for (let i = 0; i < this.outputs.length; ++i) {
                if (skipTargetIndices.indexOf(i) !== -1) {
                    continue;
                }
                // TODO(cais): Add weightedLoss, sampleWeight and mask.
                //   The following line should be weightedLoss
                const weightedLoss = this.lossFunctions[i];
                if (this.outputs.length > 1) {
                    this.metricsTensors.push([weightedLoss, i]);
                    this.metricsNames.push(this.outputNames[i] + '_loss');
                }
            }
            // Porting Note: Due to the imperative nature of the backend, we calculate
            //   the regularizer penalties in the totalLossFunction, instead of here.
        });
        const nestedMetrics = collectMetrics(args.metrics, this.outputNames);
        // TODO(cais): Add nestedWeightedMetrics.
        /**
         * Helper function used in loop below.
         */
        const appendMetric = (outputIndex, metricName, metricTensor) => {
            if (this.outputNames.length > 1) {
                metricName = this.outputNames[outputIndex] + '_' + metricName;
            }
            this.metricsNames.push(metricName);
            this.metricsTensors.push([metricTensor, outputIndex]);
        };
        nameScope('metric', () => {
            for (let i = 0; i < this.outputs.length; ++i) {
                if (skipTargetIndices.indexOf(i) !== -1) {
                    continue;
                }
                const outputMetrics = nestedMetrics[i];
                // TODO(cais): Add weights and outputWeightedMetrics.
                // TODO(cais): Add optional arg `weights` to the following function.
                const handleMetrics = (metrics) => {
                    const metricNamePrefix = '';
                    let metricName;
                    let accFn;
                    let weightedMetricFn;
                    //  TODO(cais): Use 'weights_' for weighted metrics.
                    for (const metric of metrics) {
                        if (typeof metric === 'string' &&
                            ['accuracy', 'acc', 'crossentropy', 'ce'].indexOf(metric) !==
                                -1) {
                            const outputShape = this.internalOutputShapes[i];
                            if (outputShape[outputShape.length - 1] === 1 ||
                                this.lossFunctions[i] === losses.binaryCrossentropy) {
                                // case: binary accuracy/crossentropy.
                                if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                                    accFn = Metrics.binaryAccuracy;
                                }
                                else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                                    accFn = Metrics.binaryCrossentropy;
                                }
                            }
                            else if (this.lossFunctions[i] ===
                                losses.sparseCategoricalCrossentropy) {
                                // case: categorical accuracy / crossentropy with sparse
                                // targets.
                                if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                                    accFn = Metrics.sparseCategoricalAccuracy;
                                }
                                else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                                    accFn = Metrics.sparseCategoricalCrossentropy;
                                }
                            }
                            else {
                                // case: categorical accuracy / crossentropy.
                                if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                                    accFn = Metrics.categoricalAccuracy;
                                }
                                else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                                    accFn = Metrics.categoricalCrossentropy;
                                }
                            }
                            let suffix;
                            if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                                suffix = 'acc';
                            }
                            else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                                suffix = 'ce';
                            }
                            // TODO(cais): Add weighting actually.
                            weightedMetricFn = accFn;
                            metricName = metricNamePrefix + suffix;
                        }
                        else {
                            const metricFn = Metrics.get(metric);
                            // TODO(cais): Add weighting actually.
                            weightedMetricFn = metricFn;
                            metricName =
                                metricNamePrefix + Metrics.getLossOrMetricName(metric);
                        }
                        // TODO(cais): Add weighting and masking to metricResult.
                        let metricResult;
                        nameScope(metricName, () => {
                            metricResult = weightedMetricFn;
                        });
                        appendMetric(i, metricName, metricResult);
                    }
                };
                handleMetrics(outputMetrics);
                // TODO(cais): Call handleMetrics with weights.
            }
        });
        // Porting Notes: Given the imperative backend of tfjs-core,
        //   there is no need for constructing the symbolic graph and placeholders.
        this.collectedTrainableWeights = this.trainableWeights;
    }
    /**
     * Check trainable weights count consistency.
     *
     * This will raise a warning if `this.trainableWeights` and
     * `this.collectedTrainableWeights` are inconsistent (i.e., have different
     * numbers of parameters).
     * Inconsistency will typically arise when one modifies `model.trainable`
     * without calling `model.compile()` again.
     */
    checkTrainableWeightsConsistency() {
        if (this.collectedTrainableWeights == null) {
            return;
        }
        if (this.trainableWeights.length !==
            this.collectedTrainableWeights.length) {
            console.warn('Discrepancy between trainableweights and collected trainable ' +
                'weights. Did you set `model.trainable` without calling ' +
                '`model.compile()` afterwards?');
        }
    }
    /**
     * Returns the loss value & metrics values for the model in test mode.
     *
     * Loss and metrics are specified during `compile()`, which needs to happen
     * before calls to `evaluate()`.
     *
     * Computation is done in batches.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * const result = model.evaluate(
     *     tf.ones([8, 10]), tf.ones([8, 1]), {batchSize: 4});
     * result.print();
     * ```
     *
     * @param x `tf.Tensor` of test data, or an `Array` of `tf.Tensor`s if the
     * model has multiple inputs.
     * @param y `tf.Tensor` of target data, or an `Array` of `tf.Tensor`s if the
     * model has multiple outputs.
     * @param args A `ModelEvaluateArgs`, containing optional fields.
     *
     * @return `Scalar` test loss (if the model has a single output and no
     *   metrics) or `Array` of `Scalar`s (if the model has multiple outputs
     *   and/or metrics). The attribute `model.metricsNames`
     *   will give you the display labels for the scalar outputs.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    evaluate(x, y, args = {}) {
        const batchSize = args.batchSize == null ? 32 : args.batchSize;
        checkBatchSize(batchSize);
        // TODO(cais): Standardize `config.sampleWeights` as well.
        // Validate user data.
        const checkBatchAxis = true;
        const standardizedOuts = this.standardizeUserDataXY(x, y, checkBatchAxis, batchSize);
        try {
            // TODO(cais): If uses `useLearningPhase`, set the corresponding element
            // of the input to 0.
            const ins = standardizedOuts[0].concat(standardizedOuts[1]);
            this.makeTestFunction();
            const f = this.testFunction;
            const testOuts = this.testLoop(f, ins, batchSize, args.verbose, args.steps);
            return singletonOrArray(testOuts);
        }
        finally {
            disposeNewTensors(standardizedOuts[0], x);
            disposeNewTensors(standardizedOuts[1], y);
        }
    }
    // TODO(cais): Add code snippet below once real dataset objects are
    //   available.
    /**
     * Evaluate model using a dataset object.
     *
     * Note: Unlike `evaluate()`, this method is asynchronous (`async`).
     *
     * @param dataset A dataset object. Its `iterator()` method is expected
     *   to generate a dataset iterator object, the `next()` method of which
     *   is expected to produce data batches for evaluation. The return value
     *   of the `next()` call ought to contain a boolean `done` field and a
     *   `value` field. The `value` field is expected to be an array of two
     *   `tf.Tensor`s or an array of two nested `tf.Tensor` structures. The former
     *   case is for models with exactly one input and one output (e.g.
     *   a sequential model). The latter case is for models with multiple
     *   inputs and/or multiple outputs. Of the two items in the array, the
     *   first is the input feature(s) and the second is the output target(s).
     * @param args A configuration object for the dataset-based evaluation.
     * @returns Loss and metric values as an Array of `Scalar` objects.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    async evaluateDataset(dataset, args) {
        this.makeTestFunction();
        return evaluateDataset(this, dataset, args);
    }
    /**
     * Get number of samples provided for training, evaluation or prediction.
     *
     * @param ins Input `tf.Tensor`.
     * @param batchSize Integer batch size, optional.
     * @param steps Total number of steps (batches of samples) before
     * declaring loop finished. Optional.
     * @param stepsName The public API's parameter name for `steps`.
     * @returns Number of samples provided.
     */
    checkNumSamples(ins, batchSize, steps, stepsName = 'steps') {
        let numSamples;
        if (steps != null) {
            numSamples = null;
            if (batchSize != null) {
                throw new ValueError(`If ${stepsName} is set, batchSize must be null or undefined.` +
                    `Got batchSize = ${batchSize}`);
            }
        }
        else if (ins != null) {
            if (Array.isArray(ins)) {
                numSamples = ins[0].shape[0];
            }
            else {
                numSamples = ins.shape[0];
            }
        }
        else {
            throw new ValueError(`Either the input data should have a defined shape, or ` +
                `${stepsName} shoud be specified.`);
        }
        return numSamples;
    }
    /**
     * Execute internal tensors of the model with input data feed.
     * @param inputs Input data feed. Must match the inputs of the model.
     * @param outputs Names of the output tensors to be fetched. Must match
     *   names of the SymbolicTensors that belong to the graph.
     * @returns Fetched values for `outputs`.
     */
    execute(inputs, outputs) {
        if (Array.isArray(outputs) && outputs.length === 0) {
            throw new ValueError('`outputs` is an empty Array, which is not allowed.');
        }
        const outputsIsArray = Array.isArray(outputs);
        const outputNames = (outputsIsArray ? outputs : [outputs]);
        const outputSymbolicTensors = this.retrieveSymbolicTensors(outputNames);
        // Format the input into a FeedDict.
        const feedDict = new FeedDict();
        if (inputs instanceof Tensor) {
            inputs = [inputs];
        }
        if (Array.isArray(inputs)) {
            if (inputs.length !== this.inputs.length) {
                throw new ValueError(`The number of inputs provided (${inputs.length}) ` +
                    `does not match the number of inputs of this model ` +
                    `(${this.inputs.length}).`);
            }
            for (let i = 0; i < this.inputs.length; ++i) {
                feedDict.add(this.inputs[i], inputs[i]);
            }
        }
        else {
            for (const input of this.inputs) {
                const tensorValue = inputs[input.name];
                if (tensorValue == null) {
                    throw new ValueError(`No value is provided for the model's input ${input.name}`);
                }
                feedDict.add(input, tensorValue);
            }
        }
        // Run execution.
        const executeOutputs = execute(outputSymbolicTensors, feedDict);
        return outputsIsArray ? executeOutputs : executeOutputs[0];
    }
    /**
     * Retrieve the model's internal symbolic tensors from symbolic-tensor names.
     */
    retrieveSymbolicTensors(symbolicTensorNames) {
        const outputSymbolicTensors = pyListRepeat(null, symbolicTensorNames.length);
        let outputsRemaining = symbolicTensorNames.length;
        for (const layer of this.layers) {
            const layerOutputs = Array.isArray(layer.output) ? layer.output : [layer.output];
            const layerOutputNames = layerOutputs.map(output => output.name);
            for (let i = 0; i < symbolicTensorNames.length; ++i) {
                const index = layerOutputNames.indexOf(symbolicTensorNames[i]);
                if (index !== -1) {
                    outputSymbolicTensors[i] = layerOutputs[index];
                    outputsRemaining--;
                }
                if (outputsRemaining === 0) {
                    break;
                }
            }
            if (outputsRemaining === 0) {
                break;
            }
        }
        if (outputsRemaining > 0) {
            const remainingNames = [];
            outputSymbolicTensors.forEach((tensor, i) => {
                if (tensor == null) {
                    remainingNames.push(symbolicTensorNames[i]);
                }
            });
            throw new ValueError(`Cannot find SymbolicTensors for output name(s): ` +
                `${JSON.stringify(remainingNames)}`);
        }
        return outputSymbolicTensors;
    }
    /**
     * Helper method to loop over some data in batches.
     *
     * Porting Note: Not using the functional approach in the Python equivalent
     *   due to the imperative backend.
     * Porting Note: Does not support step mode currently.
     *
     * @param ins: input data
     * @param batchSize: integer batch size.
     * @param verbose: verbosity model
     * @returns: Predictions as `tf.Tensor` (if a single output) or an `Array` of
     *   `tf.Tensor` (if multipe outputs).
     */
    predictLoop(ins, batchSize = 32, verbose = false) {
        return tfc.tidy(() => {
            const numSamples = this.checkNumSamples(ins);
            if (verbose) {
                throw new NotImplementedError('Verbose predictLoop() is not implemented yet.');
            }
            // Sample-based predictions.
            // Porting Note: Tensor currently does not support sliced assignments as
            //   in numpy, e.g., x[1:3] = y. Therefore we use concatenation while
            //   iterating over the batches.
            const batches = makeBatches(numSamples, batchSize);
            const outsBatches = this.outputs.map(output => []);
            // TODO(cais): Can the scope() be pushed down inside the for loop?
            for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
                const batchOuts = tfc.tidy(() => {
                    const batchStart = batches[batchIndex][0];
                    const batchEnd = batches[batchIndex][1];
                    // TODO(cais): Take care of the case of the last element is a flag for
                    //   training/test.
                    const insBatch = sliceArrays(ins, batchStart, batchEnd);
                    // Construct the feeds for execute();
                    const feeds = [];
                    if (Array.isArray(insBatch)) {
                        for (let i = 0; i < insBatch.length; ++i) {
                            feeds.push({ key: this.inputs[i], value: insBatch[i] });
                        }
                    }
                    else {
                        feeds.push({ key: this.inputs[0], value: insBatch });
                    }
                    const feedDict = new FeedDict(feeds);
                    return execute(this.outputs, feedDict);
                });
                batchOuts.forEach((batchOut, i) => outsBatches[i].push(batchOut));
            }
            return singletonOrArray(outsBatches.map(batches => tfc.concat(batches, 0)));
        });
    }
    /**
     * Generates output predictions for the input samples.
     *
     * Computation is done in batches.
     *
     * Note: the "step" mode of predict() is currently not supported.
     *   This is because the TensorFlow.js core backend is imperative only.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.predict(tf.ones([8, 10]), {batchSize: 4}).print();
     * ```
     *
     * @param x The input data, as a Tensor, or an `Array` of `tf.Tensor`s if
     *   the model has multiple inputs.
     * @param args A `ModelPredictArgs` object containing optional fields.
     *
     * @return Prediction results as a `tf.Tensor`(s).
     *
     * @exception ValueError In case of mismatch between the provided input data
     *   and the model's expectations, or in case a stateful model receives a
     *   number of samples that is not a multiple of the batch size.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    predict(x, args = {}) {
        const xsRank2OrHigher = ensureTensorsRank2OrHigher(x);
        checkInputData(xsRank2OrHigher, this.inputNames, this.feedInputShapes, false);
        try {
            // TODO(cais): Take care of stateful models.
            //   if (this.stateful) ...
            // TODO(cais): Take care of the learning_phase boolean flag.
            //   if (this.useLearningPhase) ...
            const batchSize = args.batchSize == null ? 32 : args.batchSize;
            checkBatchSize(batchSize);
            return this.predictLoop(xsRank2OrHigher, batchSize);
        }
        finally {
            disposeNewTensors(xsRank2OrHigher, x);
        }
    }
    /**
     * Returns predictions for a single batch of samples.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.predictOnBatch(tf.ones([8, 10])).print();
     * ```
     * @param x: Input samples, as a Tensor (for models with exactly one
     *   input) or an array of Tensors (for models with more than one input).
     * @return Tensor(s) of predictions
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    predictOnBatch(x) {
        checkInputData(x, this.inputNames, this.feedInputShapes, true);
        // TODO(cais): Take care of the learning_phase boolean flag.
        //   if (this.useLearningPhase) ...
        const batchSize = (Array.isArray(x) ? x[0] : x).shape[0];
        return this.predictLoop(x, batchSize);
    }
    standardizeUserDataXY(x, y, checkBatchAxis = true, batchSize) {
        // TODO(cais): Add sampleWeight, classWeight
        if (this.optimizer_ == null) {
            throw new RuntimeError('You must compile a model before training/testing. Use ' +
                'LayersModel.compile(modelCompileArgs).');
        }
        const outputShapes = [];
        for (let i = 0; i < this.feedOutputShapes.length; ++i) {
            const outputShape = this.feedOutputShapes[i];
            const lossFn = this.feedLossFns[i];
            if (lossFn === losses.sparseCategoricalCrossentropy) {
                outputShapes.push(outputShape.slice(0, outputShape.length - 1).concat([1]));
            }
            else {
                // Porting Note: Because of strong typing `lossFn` must be a function.
                outputShapes.push(outputShape);
            }
        }
        x = standardizeInputData(x, this.feedInputNames, this.feedInputShapes, false, 'input');
        y = standardizeInputData(y, this.feedOutputNames, outputShapes, false, 'target');
        // TODO(cais): Standardize sampleWeights & classWeights.
        checkArrayLengths(x, y, null);
        // TODO(cais): Check sampleWeights as well.
        checkLossAndTargetCompatibility(y, this.feedLossFns, this.feedOutputShapes);
        if (this.stateful && batchSize != null && batchSize > 0) {
            if (x[0].shape[0] % batchSize !== 0) {
                throw new ValueError(`In a stateful network, you should only pass inputs with a ` +
                    `number of samples that is divisible by the batch size ` +
                    `${batchSize}. Found: ${x[0].shape[0]} sample(s).`);
            }
        }
        return [x, y];
    }
    async standardizeUserData(x, y, sampleWeight, classWeight, checkBatchAxis = true, batchSize) {
        const [standardXs, standardYs] = this.standardizeUserDataXY(x, y, checkBatchAxis, batchSize);
        // TODO(cais): Handle sampleWeights.
        if (sampleWeight != null) {
            throw new Error('sample weight is not supported yet.');
        }
        let standardSampleWeights = null;
        if (classWeight != null) {
            const classWeights = standardizeClassWeights(classWeight, this.outputNames);
            standardSampleWeights = [];
            for (let i = 0; i < classWeights.length; ++i) {
                standardSampleWeights.push(await standardizeWeights(standardYs[i], null, classWeights[i]));
            }
        }
        // TODO(cais): Deal with the case of model.stateful == true.
        return [standardXs, standardYs, standardSampleWeights];
    }
    /**
     * Loop over some test data in batches.
     * @param f A Function returning a list of tensors.
     * @param ins Array of tensors to be fed to `f`.
     * @param batchSize Integer batch size or `null` / `undefined`.
     * @param verbose verbosity mode.
     * @param steps Total number of steps (batches of samples) before
     * declaring test finished. Ignored with the default value of `null` /
     * `undefined`.
     * @returns Array of Scalars.
     */
    testLoop(f, ins, batchSize, verbose = 0, steps) {
        return tfc.tidy(() => {
            const numSamples = this.checkNumSamples(ins, batchSize, steps, 'steps');
            const outs = [];
            if (verbose > 0) {
                throw new NotImplementedError('Verbose mode is not implemented yet.');
            }
            // TODO(cais): Use `indicesForConversionToDense' to prevent slow down.
            if (steps != null) {
                throw new NotImplementedError('steps mode in testLoop() is not implemented yet');
            }
            else {
                const batches = makeBatches(numSamples, batchSize);
                const indexArray = tensor1d(range(0, numSamples));
                for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
                    const batchStart = batches[batchIndex][0];
                    const batchEnd = batches[batchIndex][1];
                    const batchIds = K.sliceAlongFirstAxis(indexArray, batchStart, batchEnd - batchStart);
                    // TODO(cais): In ins, train flag can be a number, instead of an
                    //   Tensor? Do we need to handle this in tfjs-layers?
                    const insBatch = sliceArraysByIndices(ins, batchIds);
                    const batchOuts = f(insBatch);
                    if (batchIndex === 0) {
                        for (let i = 0; i < batchOuts.length; ++i) {
                            outs.push(scalar(0));
                        }
                    }
                    for (let i = 0; i < batchOuts.length; ++i) {
                        const batchOut = batchOuts[i];
                        outs[i] =
                            tfc.add(outs[i], tfc.mul(batchEnd - batchStart, batchOut));
                    }
                }
                for (let i = 0; i < outs.length; ++i) {
                    outs[i] = tfc.div(outs[i], numSamples);
                }
            }
            return outs;
        });
    }
    getDedupedMetricsNames() {
        const outLabels = this.metricsNames;
        // Rename duplicated metrics names (can happen with an output layer
        // shared among multiple dataflows).
        const dedupedOutLabels = [];
        for (let i = 0; i < outLabels.length; ++i) {
            const label = outLabels[i];
            let newLabel = label;
            if (count(outLabels, label) > 1) {
                const dupIndex = count(outLabels.slice(0, i), label);
                newLabel += `_${dupIndex}`;
            }
            dedupedOutLabels.push(newLabel);
        }
        return dedupedOutLabels;
    }
    /**
     * Creates a function that performs the following actions:
     *
     * 1. computes the losses
     * 2. sums them to get the total loss
     * 3. call the optimizer computes the gradients of the LayersModel's
     *    trainable weights w.r.t. the total loss and update the variables
     * 4. calculates the metrics
     * 5. returns the values of the losses and metrics.
     */
    makeTrainFunction() {
        return (data) => {
            const lossValues = [];
            const inputs = data.slice(0, this.inputs.length);
            const targets = data.slice(this.inputs.length, this.inputs.length + this.outputs.length);
            const sampleWeights = data.slice(this.inputs.length + this.outputs.length, this.inputs.length + this.outputs.length * 2);
            const metricsValues = [];
            // Create a function that computes the total loss based on the
            // inputs. This function is used for obtaining gradients through
            // backprop.
            const totalLossFunction = () => {
                const feeds = [];
                for (let i = 0; i < this.inputs.length; ++i) {
                    feeds.push({ key: this.inputs[i], value: inputs[i] });
                }
                const feedDict = new FeedDict(feeds);
                const outputs = execute(this.outputs, feedDict, { 'training': true });
                // TODO(cais): Take care of the case of multiple outputs from a
                //   single layer?
                let totalLoss;
                for (let i = 0; i < this.lossFunctions.length; ++i) {
                    const lossFunction = this.lossFunctions[i];
                    let loss = lossFunction(targets[i], outputs[i]);
                    if (sampleWeights[i] != null) {
                        loss = computeWeightedLoss(loss, sampleWeights[i]);
                    }
                    // TODO(cais): push Scalar instead.
                    const meanLoss = tfc.mean(loss);
                    // TODO(cais): Use a scope() instead, to avoid ownership.
                    lossValues.push(meanLoss);
                    if (i === 0) {
                        totalLoss = loss;
                    }
                    else {
                        totalLoss = tfc.add(totalLoss, loss);
                    }
                }
                // Compute the metrics.
                // TODO(cais): These should probably be calculated outside
                //   totalLossFunction to benefit speed?
                for (let i = 0; i < this.metricsTensors.length; ++i) {
                    let weightedMetric;
                    if (this.outputs.length > 1 && i < this.outputs.length) {
                        weightedMetric = lossValues[i];
                    }
                    else {
                        const metric = this.metricsTensors[i][0];
                        const outputIndex = this.metricsTensors[i][1];
                        weightedMetric =
                            tfc.mean(metric(targets[outputIndex], outputs[outputIndex]));
                    }
                    tfc.keep(weightedMetric);
                    // TODO(cais): Use a scope() instead, to avoid ownership.
                    metricsValues.push(weightedMetric);
                }
                totalLoss = tfc.mean(totalLoss);
                // Add regularizer penalties.
                this.calculateLosses().forEach(regularizerLoss => {
                    totalLoss = tfc.add(totalLoss, regularizerLoss);
                });
                return totalLoss;
            };
            const variables = this.collectedTrainableWeights.map(param => param.read());
            const returnCost = true;
            const totalLossValue = this.optimizer_.minimize(totalLossFunction, returnCost, variables);
            return [totalLossValue].concat(metricsValues);
        };
    }
    /**
     * Create a function which, when invoked with an array of `tf.Tensor`s as a
     * batch of inputs, returns the prespecified loss and metrics of the model
     * under the batch of input data.
     */
    makeTestFunction() {
        this.testFunction = (data) => {
            return tfc.tidy(() => {
                const valOutputs = [];
                let totalLoss;
                const inputs = data.slice(0, this.inputs.length);
                const targets = data.slice(this.inputs.length, this.inputs.length + this.outputs.length);
                const feeds = [];
                for (let i = 0; i < this.inputs.length; ++i) {
                    feeds.push({ key: this.inputs[i], value: inputs[i] });
                }
                const feedDict = new FeedDict(feeds);
                const outputs = execute(this.outputs, feedDict);
                // Compute total loss.
                for (let i = 0; i < this.lossFunctions.length; ++i) {
                    const lossFunction = this.lossFunctions[i];
                    // TODO(cais): Add sample weighting and replace the simple
                    // averaging.
                    const loss = tfc.mean(lossFunction(targets[i], outputs[i]));
                    if (i === 0) {
                        totalLoss = loss;
                    }
                    else {
                        totalLoss = tfc.add(totalLoss, loss);
                    }
                    valOutputs.push(totalLoss);
                }
                // Compute the metrics.
                for (let i = 0; i < this.metricsTensors.length; ++i) {
                    const metric = this.metricsTensors[i][0];
                    const outputIndex = this.metricsTensors[i][1];
                    // TODO(cais): Replace K.mean() with a proper weighting function.
                    const meanMetric = tfc.mean(metric(targets[outputIndex], outputs[outputIndex]));
                    valOutputs.push(meanMetric);
                }
                return valOutputs;
            });
        };
    }
    /**
     * Trains the model for a fixed number of epochs (iterations on a
     * dataset).
     *
     * ```js
     * const model = tf.sequential({
     *     layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * for (let i = 1; i < 5 ; ++i) {
     *   const h = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
     *       batchSize: 4,
     *       epochs: 3
     *   });
     *   console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
     * }
     * ```
     *
     * @param x `tf.Tensor` of training data, or an array of `tf.Tensor`s if the
     * model has multiple inputs. If all inputs in the model are named, you
     * can also pass a dictionary mapping input names to `tf.Tensor`s.
     * @param y `tf.Tensor` of target (label) data, or an array of `tf.Tensor`s if
     * the model has multiple outputs. If all outputs in the model are named,
     * you can also pass a dictionary mapping output names to `tf.Tensor`s.
     * @param args A `ModelFitArgs`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @exception ValueError In case of mismatch between the provided input
     * data and what the model expects.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    async fit(x, y, args = {}) {
        if (this.isTraining) {
            throw new Error('Cannot start training because another fit() call is ongoing.');
        }
        this.isTraining = true;
        let inputs;
        let targets;
        let originalInputs;
        let originalTargets;
        let inputValX;
        let inputValY;
        let valX;
        let valY;
        let sampleWeights;
        try {
            const batchSize = args.batchSize == null ? 32 : args.batchSize;
            checkBatchSize(batchSize);
            // Validate user data.
            // TODO(cais): Support sampleWeight.
            const checkBatchAxis = false;
            const standardizedOuts = await this.standardizeUserData(x, y, args.sampleWeight, args.classWeight, checkBatchAxis, batchSize);
            inputs = standardizedOuts[0];
            targets = standardizedOuts[1];
            sampleWeights = standardizedOuts[2];
            // Prepare validation data.
            let doValidation = false;
            let valIns;
            if (args.validationData != null && args.validationData.length > 0) {
                doValidation = true;
                if (args.validationData.length === 2) {
                    // config.validationData consists of valX and valY.
                    inputValX = args.validationData[0];
                    inputValY = args.validationData[1];
                }
                else if (args.validationData.length === 3) {
                    throw new NotImplementedError('validationData including sample weights is not supported yet.');
                }
                else {
                    throw new ValueError(`When passing validation data, it must contain 2 (valX, valY) ` +
                        `or 3 (valX, valY, valSampleWeight) items; ` +
                        `${args.validationData} is invalid.`);
                }
                const checkBatchAxis = true;
                const valStandardized = await this.standardizeUserData(inputValX, inputValY, null, /** Unused sample weights. */ null, /** Unused class weights. */ checkBatchAxis, batchSize);
                valX = valStandardized[0];
                valY = valStandardized[1];
                valIns = valX.concat(valY);
                // TODO(cais): Add useLearningPhase data properly.
            }
            else if (args.validationSplit != null && args.validationSplit > 0 &&
                args.validationSplit < 1) {
                doValidation = true;
                // Porting Note: In tfjs-layers, inputs[0] is always a Tensor.
                const splitAt = Math.floor(inputs[0].shape[0] * (1 - args.validationSplit));
                const originalBatchSize = inputs[0].shape[0];
                valX = sliceArrays(inputs, splitAt, originalBatchSize);
                originalInputs = inputs;
                inputs = sliceArrays(inputs, 0, splitAt);
                valY = sliceArrays(targets, splitAt, originalBatchSize);
                originalTargets = targets;
                targets = sliceArrays(targets, 0, splitAt);
                // TODO(cais): Once sampleWeights becomes available, slice it to get
                //   valSampleWeights.
                valIns = valX.concat(valY);
                // TODO(cais): Add useLearningPhase data properly.
            }
            else if (args.validationSteps != null) {
                doValidation = true;
                // TODO(cais): Add useLearningPhase.
            }
            const ins = inputs.concat(targets).concat(sampleWeights);
            this.checkTrainableWeightsConsistency();
            // TODO(cais): Handle use_learning_phase and learning_phase?
            // Porting Note: Here we see a key deviation of tfjs-layers from
            // Keras.
            //  Due to the imperative nature of tfjs-layers' backend (tfjs-core),
            //  we do not construct symbolic computation graphs to embody the
            //  training process. Instead, we define a function that performs the
            //  training action. In PyKeras, the data (inputs and targets) are fed
            //  through graph placeholders. In tfjs-layers, the data are fed as
            //  function arguments. Since the function are defined below in the
            //  scope, we don't have equivalents of PyKeras's
            //  `_make_train_funciton`.
            const trainFunction = this.makeTrainFunction();
            const outLabels = this.getDedupedMetricsNames();
            let valFunction;
            let callbackMetrics;
            if (doValidation) {
                this.makeTestFunction();
                valFunction = this.testFunction;
                callbackMetrics =
                    outLabels.slice().concat(outLabels.map(n => 'val_' + n));
            }
            else {
                valFunction = null;
                valIns = [];
                callbackMetrics = outLabels.slice();
            }
            const callbacks = standardizeCallbacks(args.callbacks, args.yieldEvery);
            const out = await this.fitLoop(trainFunction, ins, outLabels, batchSize, args.epochs, args.verbose, callbacks, valFunction, valIns, args.shuffle, callbackMetrics, args.initialEpoch, null, null);
            return out;
        }
        finally {
            this.isTraining = false;
            // Memory clean up.
            disposeNewTensors(inputs, x);
            disposeNewTensors(targets, y);
            disposeNewTensors(originalInputs, x);
            disposeNewTensors(originalTargets, y);
            disposeNewTensors(valX, inputValX);
            disposeNewTensors(valY, inputValY);
            if (sampleWeights != null) {
                tfc.dispose(sampleWeights);
            }
        }
        // TODO(cais): Add value to outLabels.
    }
    /**
     * Abstract fit function for `f(ins)`.
     * @param f A Function returning a list of tensors. For training, this
     *   function is expected to perform the updates to the variables.
     * @param ins List of tensors to be fed to `f`.
     * @param outLabels List of strings, display names of the outputs of `f`.
     * @param batchSize Integer batch size or `== null` if unknown. Default : 32.
     * @param epochs Number of times to iterate over the data. Default : 1.
     * @param verbose Verbosity mode: 0, 1, or 2. Default: 1.
     * @param callbacks List of callbacks to be called during training.
     * @param valF Function to call for validation.
     * @param valIns List of tensors to be fed to `valF`.
     * @param shuffle Whether to shuffle the data at the beginning of every
     * epoch. Default : true.
     * @param callbackMetrics List of strings, the display names of the metrics
     *   passed to the callbacks. They should be the concatenation of the
     *   display names of the outputs of `f` and the list of display names
     *   of the outputs of `valF`.
     * @param initialEpoch Epoch at which to start training (useful for
     *   resuming a previous training run). Default : 0.
     * @param stepsPerEpoch Total number of steps (batches on samples) before
     *   declaring one epoch finished and starting the next epoch. Ignored with
     *   the default value of `undefined` or `null`.
     * @param validationSteps Number of steps to run validation for (only if
     *   doing validation from data tensors). Not applicable for tfjs-layers.
     * @returns A `History` object.
     */
    async fitLoop(f, ins, outLabels, batchSize, epochs, verbose, callbacks, valF, valIns, shuffle, callbackMetrics, initialEpoch, stepsPerEpoch, validationSteps) {
        if (batchSize == null) {
            batchSize = 32;
        }
        if (epochs == null) {
            epochs = 1;
        }
        if (shuffle == null) {
            shuffle = true;
        }
        if (initialEpoch == null) {
            initialEpoch = 0;
        }
        // TODO(cais): Change const to let below when implementing validation.
        let doValidation = false;
        if (valF != null && valIns != null) {
            doValidation = true;
            // TODO(cais): verbose message.
        }
        if (validationSteps != null) {
            doValidation = true;
            if (stepsPerEpoch == null) {
                throw new ValueError('Can only use `validationSteps` when doing step-wise training, ' +
                    'i.e., `stepsPerEpoch` must be set.');
            }
        }
        const numTrainSamples = this.checkNumSamples(ins, batchSize, stepsPerEpoch, 'steps_per_epoch');
        let indexArray;
        if (numTrainSamples != null) {
            indexArray = range(0, numTrainSamples);
        }
        if (verbose == null) {
            verbose = 1;
        }
        const { callbackList, history } = configureCallbacks(callbacks, verbose, epochs, initialEpoch, numTrainSamples, stepsPerEpoch, batchSize, doValidation, callbackMetrics);
        callbackList.setModel(this);
        this.history = history;
        await callbackList.onTrainBegin();
        this.stopTraining_ = false;
        // TODO(cais): Take care of callbacks.validation_data as in PyKeras.
        // TODO(cais): Pre-convert feeds for performance as in PyKeras.
        for (let epoch = initialEpoch; epoch < epochs; ++epoch) {
            await callbackList.onEpochBegin(epoch);
            const epochLogs = {};
            if (stepsPerEpoch != null) {
                throw new NotImplementedError('stepsPerEpoch mode is not implemented yet.');
            }
            else {
                if (shuffle === 'batch') {
                    throw new NotImplementedError('batch shuffling is not implemneted'
                        + ' yet');
                }
                else if (shuffle) {
                    util.shuffle(indexArray);
                }
                // Convert the potentially shuffled indices to Tensor1D, to avoid the
                // cost of repeated creation of Array1Ds later on.
                const epochIndexArray1D = tensor1d(indexArray);
                const batches = makeBatches(numTrainSamples, batchSize);
                for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
                    const batchLogs = {};
                    await callbackList.onBatchBegin(batchIndex, batchLogs);
                    tfc.tidy(() => {
                        const batchStart = batches[batchIndex][0];
                        const batchEnd = batches[batchIndex][1];
                        const batchIds = K.sliceAlongFirstAxis(epochIndexArray1D, batchStart, batchEnd - batchStart);
                        batchLogs['batch'] = batchIndex;
                        batchLogs['size'] = batchEnd - batchStart;
                        // TODO(cais): In ins, train flag can be a number, instead of an
                        //   Tensor? Do we need to handle this in tfjs-layers?
                        const insBatch = sliceArraysByIndices(ins, batchIds);
                        const outs = f(insBatch);
                        for (let i = 0; i < outLabels.length; ++i) {
                            const label = outLabels[i];
                            const out = outs[i];
                            batchLogs[label] = out;
                            tfc.keep(out);
                            // TODO(cais): Use scope() to avoid ownership.
                        }
                        if (batchIndex === batches.length - 1) { // Last batch.
                            if (doValidation) {
                                const valOuts = this.testLoop(valF, valIns, batchSize);
                                // Porting Notes: In tfjs-layers, valOuts is always an Array.
                                for (let i = 0; i < outLabels.length; ++i) {
                                    const label = outLabels[i];
                                    const out = valOuts[i];
                                    tfc.keep(out);
                                    // TODO(cais): Use scope() to avoid ownership.
                                    epochLogs['val_' + label] = out;
                                }
                            }
                        }
                    });
                    await callbackList.onBatchEnd(batchIndex, batchLogs);
                    disposeTensorsInLogs(batchLogs);
                    if (this.stopTraining_) {
                        break;
                    }
                    // TODO(cais): return outs as list of Tensor.
                }
                epochIndexArray1D.dispose();
            }
            // TODO(cais): Run validation at the end of the epoch.
            await callbackList.onEpochEnd(epoch, epochLogs);
            if (this.stopTraining_) {
                break;
            }
        }
        await callbackList.onTrainEnd();
        await this.history.syncData();
        return this.history;
    }
    // TODO(cais): Add code snippet below when it's possible to instantiate
    //   actual dataset objects.
    /**
     * Trains the model using a dataset object.
     *
     * @param dataset A dataset object. Its `iterator()` method is expected
     *   to generate a dataset iterator object, the `next()` method of which
     *   is expected to produce data batches for training. The return value
     *   of the `next()` call ought to contain a boolean `done` field and a
     *   `value` field. The `value` field is expected to be an array of two
     *   `tf.Tensor`s or an array of two nested `tf.Tensor` structures. The former
     *   case is for models with exactly one input and one output (e.g.
     *   a sequential model). The latter case is for models with multiple
     *   inputs and/or multiple outputs.
     *   Of the two items in the array, the first is the input feature(s) and
     *   the second is the output target(s).
     * @param args A `ModelFitDatasetArgs`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    async fitDataset(dataset, args) {
        return fitDataset(this, dataset, args);
    }
    /**
     * Runs a single gradient update on a single batch of data.
     *
     * This method differs from `fit()` and `fitDataset()` in the following
     * regards:
     *   - It operates on exactly one batch of data.
     *   - It returns only the loss and metric values, instead of
     *     returning the batch-by-batch loss and metric values.
     *   - It doesn't support fine-grained options such as verbosity and
     *     callbacks.
     *
     * @param x Input data. It could be one of the following:
     *   - A `tf.Tensor`, or an Array of `tf.Tensor`s (in case the model has
     *     multiple inputs).
     *   - An Object mapping input names to corresponding `tf.Tensor` (if the
     *     model has named inputs).
     * @param y Target data. It could be either a `tf.Tensor` or multiple
     *   `tf.Tensor`s. It should be consistent with `x`.
     * @returns Training loss or losses (in case the model has
     *   multiple outputs), along with metrics (if any), as numbers.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    async trainOnBatch(x, y) {
        // TODO(cais): Support sampleWeight and classWeight.
        // TODO(cais): Support Dataset objects.
        const standardizeOut = await this.standardizeUserData(x, y);
        const inputs = standardizeOut[0];
        const targets = standardizeOut[1];
        const trainFunction = this.makeTrainFunction();
        const losses = trainFunction(inputs.concat(targets));
        const lossValues = [];
        for (const loss of losses) {
            const v = await loss.data();
            lossValues.push(v[0]);
        }
        tfc.dispose(losses);
        disposeNewTensors(standardizeOut[0], x);
        disposeNewTensors(standardizeOut[1], y);
        return singletonOrArray(lossValues);
    }
    /**
     * Extract weight values of the model.
     *
     * @param config: An instance of `io.SaveConfig`, which specifies
     * model-saving options such as whether only trainable weights are to be
     * saved.
     * @returns A `NamedTensorMap` mapping original weight names (i.e.,
     *   non-uniqueified weight names) to their values.
     */
    getNamedWeights(config) {
        const namedWeights = [];
        const trainableOnly = config != null && config.trainableOnly;
        const weights = trainableOnly ? this.trainableWeights : this.weights;
        const weightValues = this.getWeights(trainableOnly);
        for (let i = 0; i < weights.length; ++i) {
            if (trainableOnly && !weights[i].trainable) {
                // Optionally skip non-trainable weights.
                continue;
            }
            namedWeights.push({ name: weights[i].originalName, tensor: weightValues[i] });
        }
        return namedWeights;
    }
    /**
     * Setter used for force stopping of LayersModel.fit() (i.e., training).
     *
     * Example:
     *
     * ```js
     * const input = tf.input({shape: [10]});
     * const output = tf.layers.dense({units: 1}).apply(input);
     * const model = tf.model({inputs: [input], outputs: [output]});
     * model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
     * const xs = tf.ones([8, 10]);
     * const ys = tf.zeros([8, 1]);
     *
     * const history = await model.fit(xs, ys, {
     *   epochs: 10,
     *   callbacks: {
     *     onEpochEnd: async (epoch, logs) => {
     *       if (epoch === 2) {
     *         model.stopTraining = true;
     *       }
     *     }
     *   }
     * });
     *
     * // There should be only 3 values in the loss array, instead of 10
     * values,
     * // due to the stopping after 3 epochs.
     * console.log(history.history.loss);
     * ```
     */
    set stopTraining(stop) {
        this.stopTraining_ = stop;
    }
    get stopTraining() {
        return this.stopTraining_;
    }
    get optimizer() {
        return this.optimizer_;
    }
    set optimizer(optimizer) {
        if (this.optimizer_ !== optimizer) {
            this.optimizer_ = optimizer;
            this.isOptimizerOwned = false;
        }
    }
    dispose() {
        const result = super.dispose();
        if (result.refCountAfterDispose === 0 && this.optimizer != null &&
            this.isOptimizerOwned) {
            const numTensorsBeforeOptmizerDisposal = tfc.memory().numTensors;
            this.optimizer_.dispose();
            result.numDisposedVariables +=
                numTensorsBeforeOptmizerDisposal - tfc.memory().numTensors;
        }
        return result;
    }
    getLossIdentifiers() {
        let lossNames;
        if (typeof this.loss === 'string') {
            lossNames = toSnakeCase(this.loss);
        }
        else if (Array.isArray(this.loss)) {
            for (const loss of this.loss) {
                if (typeof loss !== 'string') {
                    throw new Error('Serialization of non-string loss is not supported.');
                }
            }
            lossNames = this.loss.map(name => toSnakeCase(name));
        }
        else {
            const outputNames = Object.keys(this.loss);
            lossNames = {};
            const losses = this.loss;
            for (const outputName of outputNames) {
                if (typeof losses[outputName] === 'string') {
                    lossNames[outputName] =
                        toSnakeCase(losses[outputName]);
                }
                else {
                    throw new Error('Serialization of non-string loss is not supported.');
                }
            }
        }
        return lossNames;
    }
    getMetricIdentifiers() {
        if (typeof this.metrics === 'string' ||
            typeof this.metrics === 'function') {
            return [toSnakeCase(Metrics.getLossOrMetricName(this.metrics))];
        }
        else if (Array.isArray(this.metrics)) {
            return this.metrics.map(metric => toSnakeCase(Metrics.getLossOrMetricName(metric)));
        }
        else {
            const metricsIdentifiers = {};
            for (const key in this.metrics) {
                metricsIdentifiers[key] =
                    toSnakeCase(Metrics.getLossOrMetricName(this.metrics[key]));
            }
            return metricsIdentifiers;
        }
    }
    getTrainingConfig() {
        return {
            loss: this.getLossIdentifiers(),
            metrics: this.getMetricIdentifiers(),
            optimizer_config: {
                class_name: this.optimizer.getClassName(),
                config: this.optimizer.getConfig()
            }
        };
        // TODO(cais): Add weight_metrics when they are supported.
        // TODO(cais): Add sample_weight_mode when it's supported.
        // TODO(cais): Add loss_weights when it's supported.
    }
    loadTrainingConfig(trainingConfig) {
        if (trainingConfig.weighted_metrics != null) {
            throw new Error('Loading weight_metrics is not supported yet.');
        }
        if (trainingConfig.loss_weights != null) {
            throw new Error('Loading loss_weights is not supported yet.');
        }
        if (trainingConfig.sample_weight_mode != null) {
            throw new Error('Loading sample_weight_mode is not supported yet.');
        }
        const tsConfig = convertPythonicToTs(trainingConfig.optimizer_config);
        const optimizer = deserialize(tsConfig);
        let loss;
        if (typeof trainingConfig.loss === 'string') {
            loss = toCamelCase(trainingConfig.loss);
        }
        else if (Array.isArray(trainingConfig.loss)) {
            loss = trainingConfig.loss.map(lossEntry => toCamelCase(lossEntry));
        }
        else if (trainingConfig.loss != null) {
            loss = {};
            for (const key in trainingConfig.loss) {
                loss[key] = toCamelCase(trainingConfig.loss[key]);
            }
        }
        let metrics;
        if (Array.isArray(trainingConfig.metrics)) {
            metrics = trainingConfig.metrics.map(metric => toCamelCase(metric));
        }
        else if (trainingConfig.metrics != null) {
            metrics = {};
            for (const key in trainingConfig.metrics) {
                metrics[key] = toCamelCase(trainingConfig.metrics[key]);
            }
        }
        this.compile({ loss, metrics, optimizer });
    }
    /**
     * Save the configuration and/or weights of the LayersModel.
     *
     * An `IOHandler` is an object that has a `save` method of the proper
     * signature defined. The `save` method manages the storing or
     * transmission of serialized data ("artifacts") that represent the
     * model's topology and weights onto or via a specific medium, such as
     * file downloads, local storage, IndexedDB in the web browser and HTTP
     * requests to a server. TensorFlow.js provides `IOHandler`
     * implementations for a number of frequently used saving mediums, such as
     * `tf.io.browserDownloads` and `tf.io.browserLocalStorage`. See `tf.io`
     * for more details.
     *
     * This method also allows you to refer to certain types of `IOHandler`s
     * as URL-like string shortcuts, such as 'localstorage://' and
     * 'indexeddb://'.
     *
     * Example 1: Save `model`'s topology and weights to browser [local
     * storage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage);
     * then load it back.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * console.log('Prediction from original model:');
     * model.predict(tf.ones([1, 3])).print();
     *
     * const saveResults = await model.save('localstorage://my-model-1');
     *
     * const loadedModel = await tf.loadLayersModel('localstorage://my-model-1');
     * console.log('Prediction from loaded model:');
     * loadedModel.predict(tf.ones([1, 3])).print();
     * ```
     *
     * Example 2. Saving `model`'s topology and weights to browser
     * [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API);
     * then load it back.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * console.log('Prediction from original model:');
     * model.predict(tf.ones([1, 3])).print();
     *
     * const saveResults = await model.save('indexeddb://my-model-1');
     *
     * const loadedModel = await tf.loadLayersModel('indexeddb://my-model-1');
     * console.log('Prediction from loaded model:');
     * loadedModel.predict(tf.ones([1, 3])).print();
     * ```
     *
     * Example 3. Saving `model`'s topology and weights as two files
     * (`my-model-1.json` and `my-model-1.weights.bin`) downloaded from
     * browser.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * const saveResults = await model.save('downloads://my-model-1');
     * ```
     *
     * Example 4. Send  `model`'s topology and weights to an HTTP server.
     * See the documentation of `tf.io.http` for more details
     * including specifying request parameters and implementation of the
     * server.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * const saveResults = await model.save('http://my-server/model/upload');
     * ```
     *
     * @param handlerOrURL An instance of `IOHandler` or a URL-like,
     * scheme-based string shortcut for `IOHandler`.
     * @param config Options for saving the model.
     * @returns A `Promise` of `SaveResult`, which summarizes the result of
     * the saving, such as byte sizes of the saved artifacts for the model's
     *   topology and weight values.
     *
     * @doc {heading: 'Models', subheading: 'Classes', ignoreCI: true}
     */
    async save(handlerOrURL, config) {
        if (typeof handlerOrURL === 'string') {
            const handlers = io.getSaveHandlers(handlerOrURL);
            if (handlers.length === 0) {
                throw new ValueError(`Cannot find any save handlers for URL '${handlerOrURL}'`);
            }
            else if (handlers.length > 1) {
                throw new ValueError(`Found more than one (${handlers.length}) save handlers for ` +
                    `URL '${handlerOrURL}'`);
            }
            handlerOrURL = handlers[0];
        }
        if (handlerOrURL.save == null) {
            throw new ValueError('LayersModel.save() cannot proceed because the IOHandler ' +
                'provided does not have the `save` attribute defined.');
        }
        const weightDataAndSpecs = await io.encodeWeights(this.getNamedWeights(config));
        const returnString = false;
        const unusedArg = null;
        const modelConfig = this.toJSON(unusedArg, returnString);
        const modelArtifacts = {
            modelTopology: modelConfig,
            format: LAYERS_MODEL_FORMAT_NAME,
            generatedBy: `TensorFlow.js tfjs-layers v${version}`,
            convertedBy: null,
        };
        const includeOptimizer = config == null ? false : config.includeOptimizer;
        if (includeOptimizer && this.optimizer != null) {
            modelArtifacts.trainingConfig = this.getTrainingConfig();
            const weightType = 'optimizer';
            const { data: optimizerWeightData, specs: optimizerWeightSpecs } = await io.encodeWeights(await this.optimizer.getWeights(), weightType);
            weightDataAndSpecs.specs.push(...optimizerWeightSpecs);
            weightDataAndSpecs.data = io.concatenateArrayBuffers([weightDataAndSpecs.data, optimizerWeightData]);
        }
        if (this.userDefinedMetadata != null) {
            // Check serialized size of user-defined metadata.
            const checkSize = true;
            checkUserDefinedMetadata(this.userDefinedMetadata, this.name, checkSize);
            modelArtifacts.userDefinedMetadata = this.userDefinedMetadata;
        }
        modelArtifacts.weightData = weightDataAndSpecs.data;
        modelArtifacts.weightSpecs = weightDataAndSpecs.specs;
        return handlerOrURL.save(modelArtifacts);
    }
    /**
     * Set user-defined metadata.
     *
     * The set metadata will be serialized together with the topology
     * and weights of the model during `save()` calls.
     *
     * @param setUserDefinedMetadata
     */
    setUserDefinedMetadata(userDefinedMetadata) {
        checkUserDefinedMetadata(userDefinedMetadata, this.name);
        this.userDefinedMetadata = userDefinedMetadata;
    }
    /**
     * Get user-defined metadata.
     *
     * The metadata is supplied via one of the two routes:
     *   1. By calling `setUserDefinedMetadata()`.
     *   2. Loaded during model loading (if the model is constructed
     *      via `tf.loadLayersModel()`.)
     *
     * If no user-defined metadata is available from either of the
     * two routes, this function will return `undefined`.
     */
    getUserDefinedMetadata() {
        return this.userDefinedMetadata;
    }
}
// The class name is 'Model' rather than 'LayersModel' for backwards
// compatibility since this class name shows up in the serialization format.
/** @nocollapse */
LayersModel.className = 'Model';
serialization.registerClass(LayersModel);
/**
 * A `tf.Functional` is an alias to `tf.LayersModel`.
 *
 * See also:
 *   `tf.LayersModel`, `tf.Sequential`, `tf.loadLayersModel`.
 */
/** @doc {heading: 'Models', subheading: 'Classes'} */
export class Functional extends LayersModel {
}
Functional.className = 'Functional';
serialization.registerClass(Functional);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhaW5pbmcuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvZW5naW5lL3RyYWluaW5nLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUgseUNBQXlDO0FBRXpDLE9BQU8sS0FBSyxHQUFHLE1BQU0sdUJBQXVCLENBQUM7QUFDN0MsT0FBTyxFQUFDLEVBQUUsRUFBMEQsU0FBUyxFQUFVLE1BQU0sRUFBRSxhQUFhLEVBQUUsTUFBTSxFQUFZLFFBQVEsRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUU3SyxPQUFPLEtBQUssQ0FBQyxNQUFNLHlCQUF5QixDQUFDO0FBQzdDLE9BQU8sRUFBZSxrQkFBa0IsRUFBa0Msb0JBQW9CLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUN6SCxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ3BDLE9BQU8sRUFBQyxtQkFBbUIsRUFBRSxZQUFZLEVBQUUsVUFBVSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBS3hFLE9BQU8sRUFBQyxXQUFXLEVBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUNwRCxPQUFPLEVBQUUsb0JBQW9CLEVBQWtCLE1BQU0sU0FBUyxDQUFDO0FBQy9ELE9BQU8sS0FBSyxNQUFNLE1BQU0sV0FBVyxDQUFDO0FBQ3BDLE9BQU8sS0FBSyxPQUFPLE1BQU0sWUFBWSxDQUFDO0FBQ3RDLE9BQU8sS0FBSyxVQUFVLE1BQU0sZUFBZSxDQUFDO0FBRTVDLE9BQU8sRUFBQyx3QkFBd0IsRUFBQyxNQUFNLDBCQUEwQixDQUFDO0FBQ2xFLE9BQU8sRUFBQyxLQUFLLEVBQUUsWUFBWSxFQUFFLGdCQUFnQixFQUFFLFdBQVcsRUFBRSxXQUFXLEVBQUUsTUFBTSxFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFDL0csT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLHNCQUFzQixDQUFDO0FBQ2xELE9BQU8s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Ozs7Ozs7T0FVRztJQUNILHNCQUFzQjtRQUNwQixPQUFPLElBQUksQ0FBQyxtQkFBbUIsQ0FBQztJQUNsQyxDQUFDOztBQXRyREQsb0VBQW9FO0FBQ3BFLDRFQUE0RTtBQUM1RSxrQkFBa0I7QUFDWCxxQkFBUyxHQUFHLE9BQU8sQ0FBQztBQXFyRDdCLGFBQWEsQ0FBQyxhQUFhLENBQUMsV0FBVyxDQUFDLENBQUM7QUFFekM7Ozs7O0dBS0c7QUFDSCxzREFBc0Q7QUFDdEQsTUFBTSxPQUFPLFVBQVcsU0FBUSxXQUFXOztBQUN6QixvQkFBUyxHQUFHLFlBQVksQ0FBQztBQUUzQyxhQUFhLENBQUMsYUFBYSxDQUFDLFVBQVUsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQ1xuICpcbiAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZVxuICogbGljZW5zZSB0aGF0IGNhbiBiZSBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIG9yIGF0XG4gKiBodHRwczovL29wZW5zb3VyY2Uub3JnL2xpY2Vuc2VzL01JVC5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyogT3JpZ2luYWwgU291cmNlOiBlbmdpbmUvdHJhaW5pbmcucHkgKi9cblxuaW1wb3J0ICogYXMgdGZjIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2lvLCBNb2RlbFByZWRpY3RDb25maWcgYXMgTW9kZWxQcmVkaWN0QXJncywgTmFtZWRUZW5zb3JNYXAsIE9wdGltaXplciwgU2NhbGFyLCBzY2FsYXIsIHNlcmlhbGl6YXRpb24sIFRlbnNvciwgVGVuc29yMUQsIHRlbnNvcjFkLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQgKiBhcyBLIGZyb20gJy4uL2JhY2tlbmQvdGZqc19iYWNrZW5kJztcbmltcG9ydCB7QmFzZUNhbGxiYWNrLCBjb25maWd1cmVDYWxsYmFja3MsIEhpc3RvcnksIE1vZGVsTG9nZ2luZ1ZlcmJvc2l0eSwgc3RhbmRhcmRpemVDYWxsYmFja3N9IGZyb20gJy4uL2Jhc2VfY2FsbGJhY2tzJztcbmltcG9ydCB7bmFtZVNjb3BlfSBmcm9tICcuLi9jb21tb24nO1xuaW1wb3J0IHtOb3RJbXBsZW1lbnRlZEVycm9yLCBSdW50aW1lRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4uL2Vycm9ycyc7XG5pbXBvcnQge1NoYXBlfSBmcm9tICcuLi9rZXJhc19mb3JtYXQvY29tbW9uJztcbmltcG9ydCB7TG9zc0lkZW50aWZpZXJ9IGZyb20gJy4uL2tlcmFzX2Zvcm1hdC9sb3NzX2NvbmZpZyc7XG5pbXBvcnQge09wdGltaXplclNlcmlhbGl6YXRpb259IGZyb20gJy4uL2tlcmFzX2Zvcm1hdC9vcHRpbWl6ZXJfY29uZmlnJztcbmltcG9ydCB7TWV0cmljc0lkZW50aWZpZXIsIFRyYWluaW5nQ29uZmlnfSBmcm9tICcuLi9rZXJhc19mb3JtYXQvdHJhaW5pbmdfY29uZmlnJztcbmltcG9ydCB7ZGVzZXJpYWxpemV9IGZyb20gJy4uL2xheWVycy9zZXJpYWxpemF0aW9uJztcbmltcG9ydCB7IGRpc3Bvc2VUZW5zb3JzSW5Mb2dzLCBVbnJlc29sdmVkTG9ncyB9IGZyb20gJy4uL2xvZ3MnO1xuaW1wb3J0ICogYXMgbG9zc2VzIGZyb20gJy4uL2xvc3Nlcyc7XG5pbXBvcnQgKiBhcyBNZXRyaWNzIGZyb20gJy4uL21ldHJpY3MnO1xuaW1wb3J0ICogYXMgb3B0aW1pemVycyBmcm9tICcuLi9vcHRpbWl6ZXJzJztcbmltcG9ydCB7TG9zc09yTWV0cmljRm4sIE5hbWVkVGVuc29yfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2NoZWNrVXNlckRlZmluZWRNZXRhZGF0YX0gZnJvbSAnLi4vdXNlcl9kZWZpbmVkX21ldGFkYXRhJztcbmltcG9ydCB7Y291bnQsIHB5TGlzdFJlcGVhdCwgc2luZ2xldG9uT3JBcnJheSwgdG9DYW1lbENhc2UsIHRvU25ha2VDYXNlLCB1bmlxdWV9IGZyb20gJy4uL3V0aWxzL2dlbmVyaWNfdXRpbHMnO1xuaW1wb3J0IHtwcmludFN1bW1hcnl9IGZyb20gJy4uL3V0aWxzL2xheWVyX3V0aWxzJztcbmltcG9ydCB7cmFuZ2V9IGZyb20gJy4uL3V0aWxzL21hdGhfdXRpbHMnO1xuaW1wb3J0IHtjb252ZXJ0UHl0aG9uaWNUb1RzfSBmcm9tICcuLi91dGlscy9zZXJpYWxpemF0aW9uX3V0aWxzJztcbmltcG9ydCB7TGF5ZXJWYXJpYWJsZX0gZnJvbSAnLi4vdmFyaWFibGVzJztcbmltcG9ydCB7dmVyc2lvbn0gZnJvbSAnLi4vdmVyc2lvbic7XG5cbmltcG9ydCB7Q29udGFpbmVyLCBDb250YWluZXJBcmdzfSBmcm9tICcuL2NvbnRhaW5lcic7XG5pbXBvcnQge0RhdGFzZXR9IGZyb20gJy4vZGF0YXNldF9zdHViJztcbmltcG9ydCB7ZXhlY3V0ZSwgRmVlZERpY3R9IGZyb20gJy4vZXhlY3V0b3InO1xuaW1wb3J0IHtEaXNwb3NlUmVzdWx0LCBTeW1ib2xpY1RlbnNvcn0gZnJvbSAnLi90b3BvbG9neSc7XG5pbXBvcnQge2V2YWx1YXRlRGF0YXNldCwgZml0RGF0YXNldCwgTW9kZWxFdmFsdWF0ZURhdGFzZXRBcmdzLCBNb2RlbEZpdERhdGFzZXRBcmdzfSBmcm9tICcuL3RyYWluaW5nX2RhdGFzZXQnO1xuaW1wb3J0IHtjaGVja0JhdGNoU2l6ZSwgZGlzcG9zZU5ld1RlbnNvcnMsIGVuc3VyZVRlbnNvcnNSYW5rMk9ySGlnaGVyLCBtYWtlQmF0Y2hlcywgTW9kZWxGaXRBcmdzLCBzbGljZUFycmF5cywgc2xpY2VBcnJheXNCeUluZGljZXN9IGZyb20gJy4vdHJhaW5pbmdfdGVuc29ycyc7XG5pbXBvcnQge0NsYXNzV2VpZ2h0LCBDbGFzc1dlaWdodE1hcCwgY29tcHV0ZVdlaWdodGVkTG9zcywgc3RhbmRhcmRpemVDbGFzc1dlaWdodHMsIHN0YW5kYXJkaXplV2VpZ2h0c30gZnJvbSAnLi90cmFpbmluZ191dGlscyc7XG5cbi8qKlxuICogSGVscGVyIGZ1bmN0aW9uIGZvciBwb2x5bW9ycGhpYyBpbnB1dCBkYXRhOiAxLiBzaW5nbGV0b24gVGVuc29yLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaXNEYXRhVGVuc29yKHg6IFRlbnNvcnxUZW5zb3JbXXx7W2lucHV0TmFtZTogc3RyaW5nXTogVGVuc29yfXxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAge1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcltdfSk6IGJvb2xlYW4ge1xuICByZXR1cm4geCBpbnN0YW5jZW9mIFRlbnNvcjtcbn1cblxuLyoqXG4gKiBIZWxwZXIgZnVuY3Rpb24gZm9yIHBvbHltb3JwaGljIGlucHV0IGRhdGE6IDIuIEFycmF5IG9mIFRlbnNvci5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGlzRGF0YUFycmF5KHg6IFRlbnNvcnxUZW5zb3JbXXxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB7W2lucHV0TmFtZTogc3RyaW5nXTogVGVuc29yfSk6IGJvb2xlYW4ge1xuICByZXR1cm4gQXJyYXkuaXNBcnJheSh4KTtcbn1cblxuLyoqXG4gKiBIZWxwZXIgZnVuY3Rpb24gZm9yIHBvbHltb3JwaGljIGlucHV0IGRhdGE6IDMuIFwiZGljdFwiIG9mIFRlbnNvci5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGlzRGF0YURpY3QoeDogVGVuc29yfFRlbnNvcltdfFxuICAgICAgICAgICAgICAgICAgICAgICAgICAge1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0pOiBib29sZWFuIHtcbiAgcmV0dXJuICFpc0RhdGFUZW5zb3IoeCkgJiYgIWlzRGF0YUFycmF5KHgpO1xufVxuXG4vKipcbiAqIE5vcm1hbGl6ZXMgaW5wdXRzIGFuZCB0YXJnZXRzIHByb3ZpZGVkIGJ5IHVzZXJzLlxuICogQHBhcmFtIGRhdGEgVXNlci1wcm92aWRlZCBpbnB1dCBkYXRhIChwb2x5bW9ycGhpYykuXG4gKiBAcGFyYW0gbmFtZXMgQW4gQXJyYXkgb2YgZXhwZWN0ZWQgVGVuc29yIG5hbWVzLlxuICogQHBhcmFtIHNoYXBlcyBPcHRpb25hbCBBcnJheSBvZiBleHBlY3RlZCBUZW5zb3Igc2hhcGVzLlxuICogQHBhcmFtIGNoZWNrQmF0Y2hBeGlzIFdoZXRoZXIgdG8gY2hlY2sgdGhhdCB0aGUgYmF0Y2ggYXhpcyBvZiB0aGUgYXJyYXlzXG4gKiAgIG1hdGNoICB0aGUgZXhwZWN0ZWQgdmFsdWUgZm91bmQgaW4gYHNoYXBlc2AuXG4gKiBAcGFyYW0gZXhjZXB0aW9uUHJlZml4IFN0cmluZyBwcmVmaXggdXNlZCBmb3IgZXhjZXB0aW9uIGZvcm1hdHRpbmcuXG4gKiBAcmV0dXJucyBMaXN0IG9mIHN0YW5kYXJkaXplZCBpbnB1dCBUZW5zb3JzIChvbmUgVGVuc29yIHBlciBtb2RlbCBpbnB1dCkuXG4gKiBAdGhyb3dzIFZhbHVlRXJyb3I6IGluIGNhc2Ugb2YgaW1wcm9wZXJseSBmb3JtYXR0ZWQgdXNlciBkYXRhLlxuICovXG5leHBvcnQgZnVuY3Rpb24gc3RhbmRhcmRpemVJbnB1dERhdGEoXG4gICAgZGF0YTogVGVuc29yfFRlbnNvcltdfHtbaW5wdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LCBuYW1lczogc3RyaW5nW10sXG4gICAgc2hhcGVzPzogU2hhcGVbXSwgY2hlY2tCYXRjaEF4aXMgPSB0cnVlLCBleGNlcHRpb25QcmVmaXggPSAnJyk6IFRlbnNvcltdIHtcbiAgaWYgKG5hbWVzID09IG51bGwgfHwgbmFtZXMubGVuZ3RoID09PSAwKSB7XG4gICAgLy8gQ2hlY2sgZm9yIHRoZSBjYXNlIHdoZXJlIHRoZSBtb2RlbCBleHBlY3RlZCBubyBkYXRhLCBidXQgc29tZSBkYXRhIGdvdFxuICAgIC8vIHNlbnQuXG4gICAgaWYgKGRhdGEgIT0gbnVsbCkge1xuICAgICAgbGV0IGdvdFVuZXhwZWN0ZWREYXRhID0gZmFsc2U7XG4gICAgICBpZiAoaXNEYXRhQXJyYXkoZGF0YSkgJiYgKGRhdGEgYXMgVGVuc29yW10pLmxlbmd0aCA+IDApIHtcbiAgICAgICAgZ290VW5leHBlY3RlZERhdGEgPSB0cnVlO1xuICAgICAgfSBlbHNlIGlmIChpc0RhdGFEaWN0KGRhdGEpKSB7XG4gICAgICAgIGZvciAoY29uc3Qga2V5IGluIGRhdGEpIHtcbiAgICAgICAgICBpZiAoZGF0YS5oYXNPd25Qcm9wZXJ0eShrZXkpKSB7XG4gICAgICAgICAgICBnb3RVbmV4cGVjdGVkRGF0YSA9IHRydWU7XG4gICAgICAgICAgICBicmVhaztcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIC8vIGBkYXRhYCBpcyBhIHNpbmdsZXRvbiBUZW5zb3IgaW4gdGhpcyBjYXNlLlxuICAgICAgICBnb3RVbmV4cGVjdGVkRGF0YSA9IHRydWU7XG4gICAgICB9XG4gICAgICBpZiAoZ290VW5leHBlY3RlZERhdGEpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgRXJyb3Igd2hlbiBjaGVja2luZyBtb2RlbCAke2V4Y2VwdGlvblByZWZpeH0gZXhwZWN0ZWQgbm8gZGF0YSwgYCArXG4gICAgICAgICAgICBgYnV0IGdvdCAke2RhdGF9YCk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBbXTtcbiAgfVxuICBpZiAoZGF0YSA9PSBudWxsKSB7XG4gICAgcmV0dXJuIG5hbWVzLm1hcChuYW1lID0+IG51bGwpO1xuICB9XG5cbiAgbGV0IGFycmF5czogVGVuc29yW107XG4gIGlmIChpc0RhdGFEaWN0KGRhdGEpKSB7XG4gICAgZGF0YSA9IGRhdGEgYXMge1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn07XG4gICAgYXJyYXlzID0gW107XG4gICAgZm9yIChjb25zdCBuYW1lIG9mIG5hbWVzKSB7XG4gICAgICBpZiAoZGF0YVtuYW1lXSA9PSBudWxsKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYE5vIGRhdGEgcHJvdmlkZWQgZm9yIFwiJHtuYW1lfVwiLiBOZWVkIGRhdGEgZm9yIGVhY2gga2V5IGluOiBgICtcbiAgICAgICAgICAgIGAke25hbWVzfWApO1xuICAgICAgfVxuICAgICAgYXJyYXlzLnB1c2goZGF0YVtuYW1lXSk7XG4gICAgfVxuICB9IGVsc2UgaWYgKGlzRGF0YUFycmF5KGRhdGEpKSB7XG4gICAgZGF0YSA9IGRhdGEgYXMgVGVuc29yW107XG4gICAgaWYgKGRhdGEubGVuZ3RoICE9PSBuYW1lcy5sZW5ndGgpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBFcnJvciB3aGVuIGNoZWNraW5nIG1vZGVsICR7ZXhjZXB0aW9uUHJlZml4fTogdGhlIEFycmF5IG9mIGAgK1xuICAgICAgICAgIGBUZW5zb3JzIHRoYXQgeW91IGFyZSBwYXNzaW5nIHRvIHlvdXIgbW9kZWwgaXMgbm90IHRoZSBzaXplIHRoZSBgICtcbiAgICAgICAgICBgbW9kZWwgZXhwZWN0ZWQuIEV4cGVjdGVkIHRvIHNlZSAke25hbWVzLmxlbmd0aH0gVGVuc29yKHMpLCBidXQgYCArXG4gICAgICAgICAgYGluc3RlYWQgZ290IHRoZSBmb2xsb3dpbmcgbGlzdCBvZiBUZW5zb3Iocyk6ICR7ZGF0YX1gKTtcbiAgICB9XG4gICAgYXJyYXlzID0gZGF0YTtcbiAgfSBlbHNlIHtcbiAgICBkYXRhID0gZGF0YSBhcyBUZW5zb3I7XG4gICAgaWYgKG5hbWVzLmxlbmd0aCA+IDEpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBUaGUgbW9kZWwgJHtleGNlcHRpb25QcmVmaXh9IGV4cGVjdHMgJHtuYW1lcy5sZW5ndGh9IFRlbnNvcihzKSwgYCArXG4gICAgICAgICAgYGJ1dCBvbmx5IHJlY2VpdmVkIG9uZSBUZW5zb3IuIEZvdW5kOiBUZW5zb3Igd2l0aCBzaGFwZSAke1xuICAgICAgICAgICAgICBkYXRhLnNoYXBlfWApO1xuICAgIH1cbiAgICBhcnJheXMgPSBbZGF0YV07XG4gIH1cblxuICBhcnJheXMgPSBlbnN1cmVUZW5zb3JzUmFuazJPckhpZ2hlcihhcnJheXMpO1xuXG4gIC8vIENoZWNrIHNoYXBlIGNvbXBhdGliaWxpdHkuXG4gIGlmIChzaGFwZXMgIT0gbnVsbCkge1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbmFtZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGlmIChzaGFwZXNbaV0gPT0gbnVsbCkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IGFycmF5ID0gYXJyYXlzW2ldO1xuICAgICAgaWYgKGFycmF5LnNoYXBlLmxlbmd0aCAhPT0gc2hhcGVzW2ldLmxlbmd0aCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBFcnJvciB3aGVuIGNoZWNraW5nICR7ZXhjZXB0aW9uUHJlZml4fTogZXhwZWN0ZWQgJHtuYW1lc1tpXX0gYCArXG4gICAgICAgICAgICBgdG8gaGF2ZSAke3NoYXBlc1tpXS5sZW5ndGh9IGRpbWVuc2lvbihzKS4gYnV0IGdvdCBhcnJheSB3aXRoIGAgK1xuICAgICAgICAgICAgYHNoYXBlICR7YXJyYXkuc2hhcGV9YCk7XG4gICAgICB9XG4gICAgICBmb3IgKGxldCBqID0gMDsgaiA8IHNoYXBlc1tpXS5sZW5ndGg7ICsraikge1xuICAgICAgICBpZiAoaiA9PT0gMCAmJiAhY2hlY2tCYXRjaEF4aXMpIHtcbiAgICAgICAgICAvLyBTa2lwIHRoZSBmaXJzdCAoYmF0Y2gpIGF4aXMuXG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgZGltID0gYXJyYXkuc2hhcGVbal07XG4gICAgICAgIGNvbnN0IHJlZkRpbSA9IHNoYXBlc1tpXVtqXTtcbiAgICAgICAgaWYgKHJlZkRpbSAhPSBudWxsICYmIHJlZkRpbSA+PSAwICYmIGRpbSAhPT0gcmVmRGltKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGAke2V4Y2VwdGlvblByZWZpeH0gZXhwZWN0ZWQgYSBiYXRjaCBvZiBlbGVtZW50cyB3aGVyZSBlYWNoIGAgK1xuICAgICAgICAgICAgICBgZXhhbXBsZSBoYXMgc2hhcGUgWyR7c2hhcGVzW2ldLnNsaWNlKDEsIHNoYXBlc1tpXS5sZW5ndGgpfV0gYCArXG4gICAgICAgICAgICAgIGAoaS5lLix0ZW5zb3Igc2hhcGUgWyosJHtcbiAgICAgICAgICAgICAgICAgIHNoYXBlc1tpXS5zbGljZSgxLCBzaGFwZXNbaV0ubGVuZ3RoKX1dKWAgK1xuICAgICAgICAgICAgICBgIGJ1dCB0aGUgJHtleGNlcHRpb25QcmVmaXh9IHJlY2VpdmVkIGFuIGlucHV0IHdpdGggJHtcbiAgICAgICAgICAgICAgICAgIGFycmF5LnNoYXBlWzBdfWAgK1xuICAgICAgICAgICAgICBgIGV4YW1wbGVzLCBlYWNoIHdpdGggc2hhcGUgWyR7XG4gICAgICAgICAgICAgICAgICBhcnJheS5zaGFwZS5zbGljZSgxLCBhcnJheS5zaGFwZS5sZW5ndGgpfV1gICtcbiAgICAgICAgICAgICAgYCAodGVuc29yIHNoYXBlIFske2FycmF5LnNoYXBlfV0pYCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbiAgcmV0dXJuIGFycmF5cztcbn1cblxuLyoqXG4gKiBVc2VyIGlucHV0IHZhbGlkYXRpb24gZm9yIFRlbnNvcnMuXG4gKiBAcGFyYW0gaW5wdXRzIGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzIGZvciBpbnB1dHMuXG4gKiBAcGFyYW0gdGFyZ2V0cyBgQXJyYXlgIG9mIGB0Zi5UZW5zb3JgcyBmb3IgdGFyZ2V0cy5cbiAqIEBwYXJhbSB3ZWlnaHRzIE9wdGlvbmFsIGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzIGZvciBzYW1wbGUgd2VpZ2h0cy5cbiAqIEB0aHJvd3MgVmFsdWVFcnJvcjogaW4gY2FzZSBvZiBpbmNvcnJlY3RseSBmb3JtYXR0ZWQgZGF0YS5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNoZWNrQXJyYXlMZW5ndGhzKFxuICAgIGlucHV0czogVGVuc29yW10sIHRhcmdldHM6IFRlbnNvcltdLCB3ZWlnaHRzPzogVGVuc29yW10pIHtcbiAgY29uc3Qgc2V0WCA9IHVuaXF1ZShpbnB1dHMubWFwKGlucHV0ID0+IGlucHV0LnNoYXBlWzBdKSk7XG4gIHNldFguc29ydCgpO1xuICBjb25zdCBzZXRZID0gdW5pcXVlKHRhcmdldHMubWFwKHRhcmdldCA9PiB0YXJnZXQuc2hhcGVbMF0pKTtcbiAgc2V0WS5zb3J0KCk7XG4gIC8vIFRPRE8oY2Fpcyk6IENoZWNrIGB3ZWlnaHRzYCBhcyB3ZWxsLlxuICBpZiAoc2V0WC5sZW5ndGggPiAxKSB7XG4gICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgIGBBbGwgaW5wdXQgVGVuc29ycyAoeCkgc2hvdWxkIGhhdmUgdGhlIHNhbWUgbnVtYmVyIG9mIHNhbXBsZXMuIGAgK1xuICAgICAgICBgR290IGFycmF5IHNoYXBlczogYCArXG4gICAgICAgIGAke0pTT04uc3RyaW5naWZ5KGlucHV0cy5tYXAoaW5wdXQgPT4gaW5wdXQuc2hhcGUpKX1gKTtcbiAgfVxuICBpZiAoc2V0WS5sZW5ndGggPiAxKSB7XG4gICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgIGBBbGwgdGFyZ2V0IFRlbnNvcnMgKHkpIHNob3VsZCBoYXZlIHRoZSBzYW1lIG51bWJlciBvZiBzYW1wbGVzLiBgICtcbiAgICAgICAgYEdvdCBhcnJheSBzaGFwZXM6IGAgK1xuICAgICAgICBgJHtKU09OLnN0cmluZ2lmeSh0YXJnZXRzLm1hcCh0YXJnZXQgPT4gdGFyZ2V0LnNoYXBlKSl9YCk7XG4gIH1cbiAgaWYgKHNldFgubGVuZ3RoID4gMCAmJiBzZXRZLmxlbmd0aCA+IDAgJiYgIXV0aWwuYXJyYXlzRXF1YWwoc2V0WCwgc2V0WSkpIHtcbiAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgYElucHV0IFRlbnNvcnMgc2hvdWxkIGhhdmUgdGhlIHNhbWUgbnVtYmVyIG9mIHNhbXBsZXMgYXMgdGFyZ2V0IGAgK1xuICAgICAgICBgVGVuc29ycy4gRm91bmQgJHtzZXRYWzBdfSBpbnB1dCBzYW1wbGUocykgYW5kICR7c2V0WVswXX0gdGFyZ2V0IGAgK1xuICAgICAgICBgc2FtcGxlKHMpLmApO1xuICB9XG59XG5cbi8qKlxuICogVmFsaWRhdGlvbiBvbiB0aGUgY29tcGF0aWJpbGl0eSBvZiB0YXJnZXMgYW5kIGxvc3MgZnVuY3Rpb25zLlxuICpcbiAqIFRoaXMgaGVscHMgcHJldmVudCB1c2VycyBmcm9tIHVzaW5nIGxvc3MgZnVuY3Rpb25zIGluY29ycmVjdGx5LlxuICpcbiAqIEBwYXJhbSB0YXJnZXRzIGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzIG9mIHRhcmdldHMuXG4gKiBAcGFyYW0gbG9zc0ZucyBgQXJyYXlgIG9mIGxvc3MgZnVuY3Rpb25zLlxuICogQHBhcmFtIG91dHB1dFNoYXBlcyBgQXJyYXlgIG9mIHNoYXBlcyBvZiBtb2RlbCBvdXRwdXRzLlxuICovXG5mdW5jdGlvbiBjaGVja0xvc3NBbmRUYXJnZXRDb21wYXRpYmlsaXR5KFxuICAgIHRhcmdldHM6IFRlbnNvcltdLCBsb3NzRm5zOiBMb3NzT3JNZXRyaWNGbltdLCBvdXRwdXRTaGFwZXM6IFNoYXBlW10pIHtcbiAgLy8gVE9ETyhjYWlzKTogRGVkaWNhdGVkIHRlc3QgY292ZXJhZ2U/XG4gIGNvbnN0IGtleUxvc3NlcyA9IFtcbiAgICBsb3NzZXMubWVhblNxdWFyZWRFcnJvciwgbG9zc2VzLmJpbmFyeUNyb3NzZW50cm9weSxcbiAgICBsb3NzZXMuY2F0ZWdvcmljYWxDcm9zc2VudHJvcHlcbiAgXTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCB0YXJnZXRzLmxlbmd0aDsgKytpKSB7XG4gICAgY29uc3QgeSA9IHRhcmdldHNbaV07XG4gICAgY29uc3QgbG9zcyA9IGxvc3NGbnNbaV07XG4gICAgY29uc3Qgc2hhcGUgPSBvdXRwdXRTaGFwZXNbaV07XG4gICAgaWYgKGxvc3MgPT0gbnVsbCkge1xuICAgICAgY29udGludWU7XG4gICAgfVxuICAgIGlmIChsb3NzID09PSBsb3NzZXMuY2F0ZWdvcmljYWxDcm9zc2VudHJvcHkpIHtcbiAgICAgIGlmICh5LnNoYXBlW3kuc2hhcGUubGVuZ3RoIC0gMV0gPT09IDEpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgWW91IGFyZSBwYXNzaW5nIGEgdGFyZ2V0IGFycmF5IG9mIHNoYXBlICR7eS5zaGFwZX0gd2hpbGUgdXNpbmcgYCArXG4gICAgICAgICAgICBgYSBsb3NzICdjYXRlZ29yaWNhbF9jcm9zc2VudHJvcHknLiAnY2F0ZWdvcmljYWxfY3Jvc3NlbnRyb3B5J2AgK1xuICAgICAgICAgICAgYGV4cGVjdHMgdGFyZ2V0cyB0byBiZSBiaW5hcnkgbWF0cmljZXMgKDFzIGFuZCAwcykgb2Ygc2hhcGUgYCArXG4gICAgICAgICAgICBgW3NhbXBsZXMsIGNsYXNzZXNdLmApO1xuICAgICAgICAvLyBUT0RPKGNhaXMpOiBFeGFtcGxlIGNvZGUgaW4gZXJyb3IgbWVzc2FnZS5cbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKGtleUxvc3Nlcy5pbmRleE9mKGxvc3MpICE9PSAtMSkge1xuICAgICAgY29uc3Qgc2xpY2VkWVNoYXBlID0geS5zaGFwZS5zbGljZSgxKTtcbiAgICAgIGNvbnN0IHNsaWNlZFNoYXBlID0gc2hhcGUuc2xpY2UoMSk7XG4gICAgICBmb3IgKGxldCBqID0gMDsgaiA8IHNsaWNlZFlTaGFwZS5sZW5ndGg7ICsraikge1xuICAgICAgICBjb25zdCB0YXJnZXREaW0gPSBzbGljZWRZU2hhcGVbal07XG4gICAgICAgIGNvbnN0IG91dERpbSA9IHNsaWNlZFNoYXBlW2pdO1xuICAgICAgICBpZiAob3V0RGltICE9IG51bGwgJiYgdGFyZ2V0RGltICE9PSBvdXREaW0pIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICAgYEEgdGFyZ2V0IFRlbnNvciB3aXRoIHNoYXBlICR7eS5zaGFwZX0gd2FzIHBhc3NlZCBmb3IgYW4gYCArXG4gICAgICAgICAgICAgIGBvdXRwdXQgb2Ygc2hhcGUgJHtzaGFwZX0sIHdoaWxlIHVzaW5nIGEgbG9zcyBmdW5jdGlvbiB0aGF0IGAgK1xuICAgICAgICAgICAgICBgZXhwZWN0cyB0YXJnZXRzIHRvIGhhdmUgdGhlIHNhbWUgc2hhcGUgYXMgdGhlIG91dHB1dC5gKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxufVxuXG4vKipcbiAqIENoZWNrIGlucHV0cyBwcm92aWRlZCBieSB0aGUgdXNlci5cbiAqXG4gKiBQb3J0aW5nIE5vdGU6IFRoaXMgY29ycmVzcG9uZHMgdG8gX3N0YW5kYXJkaXplX2lucHV0X2RhdGEoKSBpbiBQeXRob25cbiAqICAgS2VyYXMuIEJlY2F1c2Ugb2YgdGhlIHN0cm9uZyB0eXBpbmcgaW4gVEYuanMsIHdlIGRvIG5vdCBuZWVkIHRvIGNvbnZlcnRcbiAqICAgdGhlIGRhdGEuIFNwZWNpZmljYWxseTpcbiAqICAgMSkgaW4gUHlLZXJhcywgYGRhdGFgIGNhbiBiZSBgRGF0YUZyYW1lYCBpbnN0YW5jZXMgZnJvbSBwYW5kYXMsIGZvclxuICogICAgICBleGFtcGxlLiBXZSBkb24ndCBuZWVkIHRvIHdvcnJ5IGFib3V0IHRoYXQgaGVyZSBiZWNhdXNlIHRoZXJlIGlzIG5vXG4gKiAgICAgIHdpZGVseSBwb3B1bGFyIGphdmFzY3JpcHQvdHlwZXNkY3JpcHQgZXF1aXZhbGVudCBvZiBwYW5kYXMgKHNvIGZhcikuXG4gKiAgICAgIElmIG9uZSBiZWNvbWVzIGF2YWlsYWJsZSBpbiB0aGUgZnV0dXJlLCB3ZSBjYW4gYWRkIHN1cHBvcnQuXG4gKiAgIDIpIGluIFB5S2VyYXMsIGlucHV0cyBjYW4gYmUgUHl0aG9uIGRpY3QuIEJ1dCBoZXJlIHdlIGFyZSBzdGlwdWxhdGluZ1xuICogdGhhdCB0aGUgZGF0YSBpcyBlaXRoZXIgYSBzaW5nbGUgYHRmLlRlbnNvcmAgb3IgYW4gQXJyYXkgb2YgYHRmLlRlbnNvcmBzLiBXZVxuICogbWF5IGFkZCBzdXBwb3J0IGZvciBgT2JqZWN0YCBkYXRhIGlucHV0cyBpbiB0aGUgZnV0dXJlIHdoZW4gdGhlIG5lZWRcbiAqIGFyaXNlcy5cbiAqXG4gKiBJbnN0ZWFkLCB3ZSBwZXJmb3JtIGJhc2ljIGNoZWNrcyBmb3IgbnVtYmVyIG9mIHBhcmFtZXRlcnMgYW5kIHNoYXBlcy5cbiAqXG4gKiBAcGFyYW0gZGF0YTogVGhlIGlucHV0IGRhdGEuXG4gKiBAcGFyYW0gbmFtZXM6IE5hbWUgZm9yIHRoZSBpbnB1dHMsIGZyb20gdGhlIG1vZGVsLlxuICogQHBhcmFtIHNoYXBlczogRXhwZWN0ZWQgc2hhcGVzIGZvciB0aGUgaW5wdXQgZGF0YSwgZnJvbSB0aGUgbW9kZWwuXG4gKiBAcGFyYW0gY2hlY2tCYXRjaEF4aXM6IFdoZXRoZXIgdGhlIHNpemUgYWxvbmcgdGhlIGJhdGNoIGF4aXMgKGkuZS4sIHRoZVxuICogICBmaXJzdCBkaW1lbnNpb24pIHdpbGwgYmUgY2hlY2tlZCBmb3IgbWF0Y2hpbmcuXG4gKiBAcGFyYW0gZXhjZXB0aW9uUHJlZml4OiBFeGVjcHRpb24gcHJlZml4IG1lc3NhZ2UsIHVzZWQgaW4gZ2VuZXJhdGluZyBlcnJvclxuICogICBtZXNzYWdlcy5cbiAqIEB0aHJvd3MgVmFsdWVFcnJvcjogb24gaW5jb3JyZWN0IG51bWJlciBvZiBpbnB1dHMgb3IgbWlzbWF0Y2hlcyBpbiBzaGFwZXMuXG4gKi9cbmZ1bmN0aW9uIGNoZWNrSW5wdXREYXRhKFxuICAgIGRhdGE6IFRlbnNvcnxUZW5zb3JbXSwgbmFtZXM6IHN0cmluZ1tdLCBzaGFwZXM/OiBTaGFwZVtdLFxuICAgIGNoZWNrQmF0Y2hBeGlzID0gdHJ1ZSwgZXhjZXB0aW9uUHJlZml4ID0gJycpIHtcbiAgbGV0IGFycmF5czogVGVuc29yW107XG4gIGlmIChBcnJheS5pc0FycmF5KGRhdGEpKSB7XG4gICAgaWYgKGRhdGEubGVuZ3RoICE9PSBuYW1lcy5sZW5ndGgpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBFcnJvciB3aGVuIGNoZWNraW5nIG1vZGVsICR7ZXhjZXB0aW9uUHJlZml4fTogdGhlIEFycmF5IG9mIGAgK1xuICAgICAgICAgIGBUZW5zb3JzIHRoYXQgeW91IGFyZSBwYXNzaW5nIHRvIHlvdXIgbW9kZWwgaXMgbm90IHRoZSBzaXplIHRoZSBgICtcbiAgICAgICAgICBgdGhlIG1vZGVsIGV4cGVjdGVkLiBFeHBlY3RlZCB0byBzZWUgJHtuYW1lcy5sZW5ndGh9IFRlbnNvcihzKSxgICtcbiAgICAgICAgICBgIGJ1dCBpbnN0ZWFkIGdvdCAke2RhdGEubGVuZ3RofSBUZW5zb3JzKHMpLmApO1xuICAgIH1cbiAgICBhcnJheXMgPSBkYXRhO1xuICB9IGVsc2Uge1xuICAgIGlmIChuYW1lcy5sZW5ndGggPiAxKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgVGhlIG1vZGVsIGV4cGVjdHMgJHtuYW1lcy5sZW5ndGh9ICR7ZXhjZXB0aW9uUHJlZml4fSBUZW5zb3JzLCBgICtcbiAgICAgICAgICBgYnV0IG9ubHkgcmVjZWl2ZWQgb25lIFRlbnNvci4gRm91bmQ6IGFycmF5IHdpdGggc2hhcGUgYCArXG4gICAgICAgICAgYCR7SlNPTi5zdHJpbmdpZnkoZGF0YS5zaGFwZSl9LmApO1xuICAgIH1cbiAgICBhcnJheXMgPSBbZGF0YV07XG4gIH1cblxuICBpZiAoc2hhcGVzICE9IG51bGwpIHtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5hbWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBpZiAoc2hhcGVzW2ldID09IG51bGwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBjb25zdCBhcnJheSA9IGFycmF5c1tpXTtcbiAgICAgIGlmIChhcnJheS5zaGFwZS5sZW5ndGggIT09IHNoYXBlc1tpXS5sZW5ndGgpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgRXJyb3Igd2hlbiBjaGVja2luZyAke2V4Y2VwdGlvblByZWZpeH06IGV4cGVjdGVkICR7bmFtZXNbaV19IGAgK1xuICAgICAgICAgICAgYHRvIGhhdmUgJHtzaGFwZXNbaV0ubGVuZ3RofSBkaW1lbnNpb24ocyksIGJ1dCBnb3QgYXJyYXkgd2l0aCBgICtcbiAgICAgICAgICAgIGBzaGFwZSAke0pTT04uc3RyaW5naWZ5KGFycmF5LnNoYXBlKX1gKTtcbiAgICAgIH1cbiAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgc2hhcGVzW2ldLmxlbmd0aDsgKytqKSB7XG4gICAgICAgIGlmIChqID09PSAwICYmICFjaGVja0JhdGNoQXhpcykge1xuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGRpbSA9IGFycmF5LnNoYXBlW2pdO1xuICAgICAgICBjb25zdCByZWZEaW0gPSBzaGFwZXNbaV1bal07XG4gICAgICAgIGlmIChyZWZEaW0gIT0gbnVsbCkge1xuICAgICAgICAgIGlmIChyZWZEaW0gIT09IGRpbSkge1xuICAgICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgICAgYEVycm9yIHdoZW4gY2hlY2tpbmcgJHtleGNlcHRpb25QcmVmaXh9OiBleHBlY3RlZCBgICtcbiAgICAgICAgICAgICAgICBgJHtuYW1lc1tpXX0gdG8gaGF2ZSBzaGFwZSAke0pTT04uc3RyaW5naWZ5KHNoYXBlc1tpXSl9IGJ1dCBgICtcbiAgICAgICAgICAgICAgICBgZ290IGFycmF5IHdpdGggc2hhcGUgJHtKU09OLnN0cmluZ2lmeShhcnJheS5zaGFwZSl9LmApO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxufVxuXG4vKipcbiAqIE1hcHMgbWV0cmljIGZ1bmN0aW9ucyB0byBtb2RlbCBvdXRwdXRzLlxuICogQHBhcmFtIG1ldHJpY3MgQW4gc2hvcnRjdXQgc3RyaW5ncyBuYW1lLCBtZXRyaWMgZnVuY3Rpb24sIGBBcnJheWAgb3IgZGljdFxuICogICAoYE9iamVjdGApIG9mIG1ldHJpYyBmdW5jdGlvbnMuXG4gKiBAcGFyYW0gb3V0cHV0TmFtZXMgQW4gYEFycmF5YCBvZiB0aGUgbmFtZXMgb2YgbW9kZWwgb3V0cHV0cy5cbiAqIEByZXR1cm5zIEFuIGBBcnJheWAgKG9uZSBlbnRyeSBwZXIgbW9kZWwgb3V0cHV0KSBvZiBgQXJyYXlgIG9mIG1ldHJpY1xuICogICBmdW5jdGlvbnMuIEZvciBpbnN0YW5jZSwgaWYgdGhlIG1vZGVsIGhhcyAyIG91dHB1dHMsIGFuZCBmb3IgdGhlIGZpcnN0XG4gKiAgIG91dHB1dCB3ZSB3YW50IHRvIGNvbXB1dGUgYGJpbmFyeUFjY3VyYWN5YCBhbmQgYGJpbmFyeUNyb3NzZW50cm9weWAsXG4gKiAgIGFuZCBqdXN0IGBiaW5hcnlBY2N1cmFjeWAgZm9yIHRoZSBzZWNvbmQgb3V0cHV0LCB0aGUgYEFycmF5YCB3b3VsZCBsb29rXG4gKiAgIGxpa2U6XG4gKiAgICAgYFtbYmluYXJ5QWNjdXJhY3ksIGJpbmFyeUNyb3NzZW50cm9weV0sICBbYmluYXJ5QWNjdXJhY3ldXWBcbiAqIEB0aHJvd3MgVHlwZUVycm9yOiBpbmNvbXBhdGlibGUgbWV0cmljcyBmb3JtYXQuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjb2xsZWN0TWV0cmljcyhcbiAgICBtZXRyaWNzOiBzdHJpbmd8TG9zc09yTWV0cmljRm58QXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPnxcbiAgICB7W291dHB1dE5hbWU6IHN0cmluZ106IHN0cmluZyB8IExvc3NPck1ldHJpY0ZufSxcbiAgICBvdXRwdXROYW1lczogc3RyaW5nW10pOiBBcnJheTxBcnJheTxzdHJpbmd8TG9zc09yTWV0cmljRm4+PiB7XG4gIGlmIChtZXRyaWNzID09IG51bGwgfHwgQXJyYXkuaXNBcnJheShtZXRyaWNzKSAmJiBtZXRyaWNzLmxlbmd0aCA9PT0gMCkge1xuICAgIHJldHVybiBvdXRwdXROYW1lcy5tYXAobmFtZSA9PiBbXSk7XG4gIH1cblxuICBsZXQgd3JhcHBlZE1ldHJpY3M6IEFycmF5PHN0cmluZ3xMb3NzT3JNZXRyaWNGbj58XG4gICAgICB7W291dHB1dE5hbWU6IHN0cmluZ106IHN0cmluZyB8IExvc3NPck1ldHJpY0ZufTtcbiAgaWYgKHR5cGVvZiBtZXRyaWNzID09PSAnc3RyaW5nJyB8fCB0eXBlb2YgbWV0cmljcyA9PT0gJ2Z1bmN0aW9uJykge1xuICAgIHdyYXBwZWRNZXRyaWNzID0gW21ldHJpY3NdO1xuICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkobWV0cmljcykgfHwgdHlwZW9mIG1ldHJpY3MgPT09ICdvYmplY3QnKSB7XG4gICAgd3JhcHBlZE1ldHJpY3MgPSBtZXRyaWNzIGFzIEFycmF5PHN0cmluZ3xMb3NzT3JNZXRyaWNGbj58XG4gICAgICAgIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogc3RyaW5nfSB8IHtbb3V0cHV0TmFtZTogc3RyaW5nXTogTG9zc09yTWV0cmljRm59O1xuICB9IGVsc2Uge1xuICAgIHRocm93IG5ldyBUeXBlRXJyb3IoXG4gICAgICAgICdUeXBlIG9mIG1ldHJpY3MgYXJndW1lbnQgbm90IHVuZGVyc3Rvb2QuIEV4cGVjdGVkIGFuIHN0cmluZywnICtcbiAgICAgICAgYGZ1bmN0aW9uLCBBcnJheSwgb3IgT2JqZWN0LCBmb3VuZDogJHttZXRyaWNzfWApO1xuICB9XG5cbiAgaWYgKEFycmF5LmlzQXJyYXkod3JhcHBlZE1ldHJpY3MpKSB7XG4gICAgLy8gV2UgdGhlbiBhcHBseSBhbGwgbWV0cmljcyB0byBhbGwgb3V0cHV0cy5cbiAgICByZXR1cm4gb3V0cHV0TmFtZXMubWFwKFxuICAgICAgICBuYW1lID0+IHdyYXBwZWRNZXRyaWNzIGFzIEFycmF5PHN0cmluZ3xMb3NzT3JNZXRyaWNGbj4pO1xuICB9IGVsc2Uge1xuICAgIC8vIEluIHRoaXMgY2FzZSwgbWV0cmljcyBpcyBhIGRpY3QuXG4gICAgY29uc3QgbmVzdGVkTWV0cmljczogQXJyYXk8QXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPj4gPSBbXTtcbiAgICBmb3IgKGNvbnN0IG5hbWUgb2Ygb3V0cHV0TmFtZXMpIHtcbiAgICAgIGxldCBvdXRwdXRNZXRyaWNzOiBzdHJpbmd8TG9zc09yTWV0cmljRm58QXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPiA9XG4gICAgICAgICAgd3JhcHBlZE1ldHJpY3MuaGFzT3duUHJvcGVydHkobmFtZSkgPyB3cmFwcGVkTWV0cmljc1tuYW1lXSA6IFtdO1xuICAgICAgaWYgKCFBcnJheS5pc0FycmF5KG91dHB1dE1ldHJpY3MpKSB7XG4gICAgICAgIG91dHB1dE1ldHJpY3MgPSBbb3V0cHV0TWV0cmljc107XG4gICAgICB9XG4gICAgICBuZXN0ZWRNZXRyaWNzLnB1c2gob3V0cHV0TWV0cmljcyk7XG4gICAgfVxuICAgIHJldHVybiBuZXN0ZWRNZXRyaWNzO1xuICB9XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgTW9kZWxFdmFsdWF0ZUFyZ3Mge1xuICAvKipcbiAgICogQmF0Y2ggc2l6ZSAoSW50ZWdlcikuIElmIHVuc3BlY2lmaWVkLCBpdCB3aWxsIGRlZmF1bHQgdG8gMzIuXG4gICAqL1xuICBiYXRjaFNpemU/OiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIFZlcmJvc2l0eSBtb2RlLlxuICAgKi9cbiAgdmVyYm9zZT86IE1vZGVsTG9nZ2luZ1ZlcmJvc2l0eTtcblxuICAvKipcbiAgICogVGVuc29yIG9mIHdlaWdodHMgdG8gd2VpZ2h0IHRoZSBjb250cmlidXRpb24gb2YgZGlmZmVyZW50IHNhbXBsZXMgdG8gdGhlXG4gICAqIGxvc3MgYW5kIG1ldHJpY3MuXG4gICAqL1xuICBzYW1wbGVXZWlnaHQ/OiBUZW5zb3I7XG5cbiAgLyoqXG4gICAqIGludGVnZXI6IHRvdGFsIG51bWJlciBvZiBzdGVwcyAoYmF0Y2hlcyBvZiBzYW1wbGVzKVxuICAgKiBiZWZvcmUgZGVjbGFyaW5nIHRoZSBldmFsdWF0aW9uIHJvdW5kIGZpbmlzaGVkLiBJZ25vcmVkIHdpdGggdGhlIGRlZmF1bHRcbiAgICogdmFsdWUgb2YgYHVuZGVmaW5lZGAuXG4gICAqL1xuICBzdGVwcz86IG51bWJlcjtcbn1cblxuLyoqXG4gKiBDb25maWd1cmF0aW9uIGZvciBjYWxscyB0byBgTGF5ZXJzTW9kZWwuY29tcGlsZSgpYC5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBNb2RlbENvbXBpbGVBcmdzIHtcbiAgLyoqXG4gICAqIEFuIGluc3RhbmNlIG9mIGB0Zi50cmFpbi5PcHRpbWl6ZXJgIG9yIGEgc3RyaW5nIG5hbWUgZm9yIGFuIE9wdGltaXplci5cbiAgICovXG4gIG9wdGltaXplcjogc3RyaW5nfE9wdGltaXplcjtcblxuICAvKipcbiAgICogT2JqZWN0IGZ1bmN0aW9uKHMpIG9yIG5hbWUocykgb2Ygb2JqZWN0IGZ1bmN0aW9uKHMpLlxuICAgKiBJZiB0aGUgbW9kZWwgaGFzIG11bHRpcGxlIG91dHB1dHMsIHlvdSBjYW4gdXNlIGEgZGlmZmVyZW50IGxvc3NcbiAgICogb24gZWFjaCBvdXRwdXQgYnkgcGFzc2luZyBhIGRpY3Rpb25hcnkgb3IgYW4gQXJyYXkgb2YgbG9zc2VzLlxuICAgKiBUaGUgbG9zcyB2YWx1ZSB0aGF0IHdpbGwgYmUgbWluaW1pemVkIGJ5IHRoZSBtb2RlbCB3aWxsIHRoZW4gYmUgdGhlIHN1bVxuICAgKiBvZiBhbGwgaW5kaXZpZHVhbCBsb3NzZXMuXG4gICAqL1xuICBsb3NzOiBzdHJpbmd8c3RyaW5nW118e1tvdXRwdXROYW1lOiBzdHJpbmddOiBzdHJpbmd9fExvc3NPck1ldHJpY0ZufFxuICAgICAgTG9zc09yTWV0cmljRm5bXXx7W291dHB1dE5hbWU6IHN0cmluZ106IExvc3NPck1ldHJpY0ZufTtcblxuICAvKipcbiAgICogTGlzdCBvZiBtZXRyaWNzIHRvIGJlIGV2YWx1YXRlZCBieSB0aGUgbW9kZWwgZHVyaW5nIHRyYWluaW5nIGFuZCB0ZXN0aW5nLlxuICAgKiBUeXBpY2FsbHkgeW91IHdpbGwgdXNlIGBtZXRyaWNzPVsnYWNjdXJhY3knXWAuXG4gICAqIFRvIHNwZWNpZnkgZGlmZmVyZW50IG1ldHJpY3MgZm9yIGRpZmZlcmVudCBvdXRwdXRzIG9mIGEgbXVsdGktb3V0cHV0XG4gICAqIG1vZGVsLCB5b3UgY291bGQgYWxzbyBwYXNzIGEgZGljdGlvbmFyeS5cbiAgICovXG4gIG1ldHJpY3M/OiBzdHJpbmd8TG9zc09yTWV0cmljRm58QXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPnxcbiAgICAgIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogc3RyaW5nIHwgTG9zc09yTWV0cmljRm59O1xuXG4gIC8vIFRPRE8oY2Fpcyk6IEFkZCBsb3NzV2VpZ2h0cywgc2FtcGxlV2VpZ2h0TW9kZSwgd2VpZ2h0ZWRNZXRyaWNzLCBhbmRcbiAgLy8gICB0YXJnZXRUZW5zb3JzLlxufVxuXG5jb25zdCBMQVlFUlNfTU9ERUxfRk9STUFUX05BTUUgPSAnbGF5ZXJzLW1vZGVsJztcblxuLyoqXG4gKiBBIGB0Zi5MYXllcnNNb2RlbGAgaXMgYSBkaXJlY3RlZCwgYWN5Y2xpYyBncmFwaCBvZiBgdGYuTGF5ZXJgcyBwbHVzIG1ldGhvZHNcbiAqIGZvciB0cmFpbmluZywgZXZhbHVhdGlvbiwgcHJlZGljdGlvbiBhbmQgc2F2aW5nLlxuICpcbiAqIGB0Zi5MYXllcnNNb2RlbGAgaXMgdGhlIGJhc2ljIHVuaXQgb2YgdHJhaW5pbmcsIGluZmVyZW5jZSBhbmQgZXZhbHVhdGlvbiBpblxuICogVGVuc29yRmxvdy5qcy4gVG8gY3JlYXRlIGEgYHRmLkxheWVyc01vZGVsYCwgdXNlIGB0Zi5MYXllcnNNb2RlbGAuXG4gKlxuICogU2VlIGFsc286XG4gKiAgIGB0Zi5TZXF1ZW50aWFsYCwgYHRmLmxvYWRMYXllcnNNb2RlbGAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAqL1xuZXhwb3J0IGNsYXNzIExheWVyc01vZGVsIGV4dGVuZHMgQ29udGFpbmVyIGltcGxlbWVudHMgdGZjLkluZmVyZW5jZU1vZGVsIHtcbiAgLy8gVGhlIGNsYXNzIG5hbWUgaXMgJ01vZGVsJyByYXRoZXIgdGhhbiAnTGF5ZXJzTW9kZWwnIGZvciBiYWNrd2FyZHNcbiAgLy8gY29tcGF0aWJpbGl0eSBzaW5jZSB0aGlzIGNsYXNzIG5hbWUgc2hvd3MgdXAgaW4gdGhlIHNlcmlhbGl6YXRpb24gZm9ybWF0LlxuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdNb2RlbCc7XG4gIHByb3RlY3RlZCBvcHRpbWl6ZXJfOiBPcHRpbWl6ZXI7XG4gIC8vIFdoZXRoZXIgdGhlIG1vZGVsIGluc3RhbmNlIG93bnMgdGhlIG9wdGltaXplcjogYHRydWVgIGlmIGFuZCBvbmx5IGlmXG4gIC8vIGBvcHRpbWl6ZXJgIGlzIGNyZWF0ZWQgZnJvbSBhIHN0cmluZyBwYXJhbWV0ZXIgZHVyaW5nIGBjb21waWxlKClgIGNhbGwuXG4gIHByb3RlY3RlZCBpc09wdGltaXplck93bmVkOiBib29sZWFuO1xuXG4gIGxvc3M6IHN0cmluZ3xzdHJpbmdbXXx7W291dHB1dE5hbWU6IHN0cmluZ106IHN0cmluZ318TG9zc09yTWV0cmljRm58XG4gICAgICBMb3NzT3JNZXRyaWNGbltdfHtbb3V0cHV0TmFtZTogc3RyaW5nXTogTG9zc09yTWV0cmljRm59O1xuICBsb3NzRnVuY3Rpb25zOiBMb3NzT3JNZXRyaWNGbltdO1xuXG4gIC8vIFRPRE8oY2Fpcyk6IFRoZXNlIHByaXZhdGUgdmFyaWFibGVzIHNob3VsZCBwcm9iYWJseSBub3QgaGF2ZSB0aGUgc3RyaW5nXG4gIC8vICAgJ2ZlZWQnIGluIHRoZWlyIG5hbWVzLCBiZWNhdXNlIHdlIGFyZSBub3QgZGVhbGluZyB3aXRoIGEgc3ltYm9saWNcbiAgLy8gICBiYWNrZW5kLlxuICBwcml2YXRlIGZlZWRPdXRwdXRTaGFwZXM6IFNoYXBlW107XG4gIHByaXZhdGUgZmVlZExvc3NGbnM6IExvc3NPck1ldHJpY0ZuW107XG4gIHByaXZhdGUgY29sbGVjdGVkVHJhaW5hYmxlV2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdO1xuICBwcml2YXRlIHRlc3RGdW5jdGlvbjogKGRhdGE6IFRlbnNvcltdKSA9PiBTY2FsYXJbXTtcbiAgaGlzdG9yeTogSGlzdG9yeTtcblxuICAvLyBBIHB1YmxpYyBwcm9wZXJ0eSB0aGF0IGNhbiBiZSBzZXQgYnkgQ2FsbGJhY2tzIHRvIG9yZGVyIGVhcmx5IHN0b3BwaW5nXG4gIC8vIGR1cmluZyBgZml0KClgIGNhbGxzLlxuICBwcm90ZWN0ZWQgc3RvcFRyYWluaW5nXzogYm9vbGVhbjtcbiAgcHJvdGVjdGVkIGlzVHJhaW5pbmc6IGJvb2xlYW47XG5cbiAgbWV0cmljczogc3RyaW5nfExvc3NPck1ldHJpY0ZufEFycmF5PHN0cmluZ3xMb3NzT3JNZXRyaWNGbj58XG4gICAgICB7W291dHB1dE5hbWU6IHN0cmluZ106IHN0cmluZyB8IExvc3NPck1ldHJpY0ZufTtcbiAgbWV0cmljc05hbWVzOiBzdHJpbmdbXTtcbiAgLy8gUG9ydGluZyBOb3RlOiBgbWV0cmljc190ZW5zb3JzYCBpbiBQeUtlcmFzIGlzIGEgc3ltYm9saWMgdGVuc29yLiBCdXQgZ2l2ZW5cbiAgLy8gICB0aGUgaW1wZXJhdGl2ZSBuYXR1cmUgb2YgdGZqcy1jb3JlLCBgbWV0cmljc1RlbnNvcnNgIGlzIGFcbiAgLy8gICBUeXBlU2NyaXB0IGZ1bmN0aW9uIGhlcmUuXG4gIC8vICAgQWxzbyBub3RlIHRoYXQgZHVlIHRvIHRoZSBpbXBlcmF0aXZlIG5hdHVyZSBvZiB0ZmpzLWNvcmUsIGBtZXRyaWNzVGVuc29yYFxuICAvLyAgIGhlcmUgbmVlZHMgYW4gb3V0cHV0IGluZGV4IHRvIGtlZXAgdHJhY2sgb2Ygd2hpY2ggb3V0cHV0IG9mIHRoZVxuICAvLyAgIExheWVyc01vZGVsIGEgbWV0cmljIGJlbG9uZ3MgdG8uIFRoaXMgaXMgdW5saWtlIGBtZXRyaWNzX3RlbnNvcnNgIGluXG4gIC8vICAgUHlLZXJhcywgd2hpY2ggaXMgYSBgbGlzdGAgb2Ygc3ltYm9saWMgdGVuc29ycywgZWFjaCBvZiB3aGljaCBoYXNcbiAgLy8gICBpbXBsaWNpdCBcImtub3dsZWRnZVwiIG9mIHRoZSBvdXRwdXRzIGl0IGRlcGVuZHMgb24uXG4gIG1ldHJpY3NUZW5zb3JzOiBBcnJheTxbTG9zc09yTWV0cmljRm4sIG51bWJlcl0+O1xuXG4gIC8vIFVzZXIgZGVmaW5kIG1ldGFkYXRhIChpZiBhbnkpLlxuICBwcml2YXRlIHVzZXJEZWZpbmVkTWV0YWRhdGE6IHt9O1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IENvbnRhaW5lckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzKTtcbiAgICB0aGlzLmlzVHJhaW5pbmcgPSBmYWxzZTtcbiAgfVxuXG4gIC8qKlxuICAgKiBQcmludCBhIHRleHQgc3VtbWFyeSBvZiB0aGUgbW9kZWwncyBsYXllcnMuXG4gICAqXG4gICAqIFRoZSBzdW1tYXJ5IGluY2x1ZGVzXG4gICAqIC0gTmFtZSBhbmQgdHlwZSBvZiBhbGwgbGF5ZXJzIHRoYXQgY29tcHJpc2UgdGhlIG1vZGVsLlxuICAgKiAtIE91dHB1dCBzaGFwZShzKSBvZiB0aGUgbGF5ZXJzXG4gICAqIC0gTnVtYmVyIG9mIHdlaWdodCBwYXJhbWV0ZXJzIG9mIGVhY2ggbGF5ZXJcbiAgICogLSBJZiB0aGUgbW9kZWwgaGFzIG5vbi1zZXF1ZW50aWFsLWxpa2UgdG9wb2xvZ3ksIHRoZSBpbnB1dHMgZWFjaCBsYXllclxuICAgKiAgIHJlY2VpdmVzXG4gICAqIC0gVGhlIHRvdGFsIG51bWJlciBvZiB0cmFpbmFibGUgYW5kIG5vbi10cmFpbmFibGUgcGFyYW1ldGVycyBvZiB0aGUgbW9kZWwuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGlucHV0MSA9IHRmLmlucHV0KHtzaGFwZTogWzEwXX0pO1xuICAgKiBjb25zdCBpbnB1dDIgPSB0Zi5pbnB1dCh7c2hhcGU6IFsyMF19KTtcbiAgICogY29uc3QgZGVuc2UxID0gdGYubGF5ZXJzLmRlbnNlKHt1bml0czogNH0pLmFwcGx5KGlucHV0MSk7XG4gICAqIGNvbnN0IGRlbnNlMiA9IHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDh9KS5hcHBseShpbnB1dDIpO1xuICAgKiBjb25zdCBjb25jYXQgPSB0Zi5sYXllcnMuY29uY2F0ZW5hdGUoKS5hcHBseShbZGVuc2UxLCBkZW5zZTJdKTtcbiAgICogY29uc3Qgb3V0cHV0ID1cbiAgICogICAgIHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDMsIGFjdGl2YXRpb246ICdzb2Z0bWF4J30pLmFwcGx5KGNvbmNhdCk7XG4gICAqXG4gICAqIGNvbnN0IG1vZGVsID0gdGYubW9kZWwoe2lucHV0czogW2lucHV0MSwgaW5wdXQyXSwgb3V0cHV0czogb3V0cHV0fSk7XG4gICAqIG1vZGVsLnN1bW1hcnkoKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSBsaW5lTGVuZ3RoIEN1c3RvbSBsaW5lIGxlbmd0aCwgaW4gbnVtYmVyIG9mIGNoYXJhY3RlcnMuXG4gICAqIEBwYXJhbSBwb3NpdGlvbnMgQ3VzdG9tIHdpZHRocyBvZiBlYWNoIG9mIHRoZSBjb2x1bW5zLCBhcyBlaXRoZXJcbiAgICogICBmcmFjdGlvbnMgb2YgYGxpbmVMZW5ndGhgIChlLmcuLCBgWzAuNSwgMC43NSwgMV1gKSBvciBhYnNvbHV0ZSBudW1iZXJcbiAgICogICBvZiBjaGFyYWN0ZXJzIChlLmcuLCBgWzMwLCA1MCwgNjVdYCkuIEVhY2ggbnVtYmVyIGNvcnJlc3BvbmRzIHRvXG4gICAqICAgcmlnaHQtbW9zdCAoaS5lLiwgZW5kaW5nKSBwb3NpdGlvbiBvZiBhIGNvbHVtbi5cbiAgICogQHBhcmFtIHByaW50Rm4gQ3VzdG9tIHByaW50IGZ1bmN0aW9uLiBDYW4gYmUgdXNlZCB0byByZXBsYWNlIHRoZSBkZWZhdWx0XG4gICAqICAgYGNvbnNvbGUubG9nYC4gRm9yIGV4YW1wbGUsIHlvdSBjYW4gdXNlIGB4ID0+IHt9YCB0byBtdXRlIHRoZSBwcmludGVkXG4gICAqICAgbWVzc2FnZXMgaW4gdGhlIGNvbnNvbGUuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBzdW1tYXJ5KFxuICAgICAgbGluZUxlbmd0aD86IG51bWJlciwgcG9zaXRpb25zPzogbnVtYmVyW10sXG4gICAgICBwcmludEZuOlxuICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgIChtZXNzYWdlPzogYW55LCAuLi5vcHRpb25hbFBhcmFtczogYW55W10pID0+IHZvaWQgPSBjb25zb2xlLmxvZykge1xuICAgIGlmICghdGhpcy5idWlsdCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYFRoaXMgbW9kZWwgaGFzIG5ldmVyIGJlZW4gY2FsbGVkLCB0aHVzIGl0cyB3ZWlnaHRzIGhhdmUgbm90IGJlZW4gYCArXG4gICAgICAgICAgYGNyZWF0ZWQgeWV0LiBTbyBubyBzdW1tYXJ5IGNhbiBiZSBkaXNwbGF5ZWQuIEJ1aWxkIHRoZSBtb2RlbCBgICtcbiAgICAgICAgICBgZmlyc3QgKGUuZy4sIGJ5IGNhbGxpbmcgaXQgb24gc29tZSB0ZXN0IGRhdGEpLmApO1xuICAgIH1cbiAgICBwcmludFN1bW1hcnkodGhpcywgbGluZUxlbmd0aCwgcG9zaXRpb25zLCBwcmludEZuKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb25maWd1cmVzIGFuZCBwcmVwYXJlcyB0aGUgbW9kZWwgZm9yIHRyYWluaW5nIGFuZCBldmFsdWF0aW9uLiAgQ29tcGlsaW5nXG4gICAqIG91dGZpdHMgdGhlIG1vZGVsIHdpdGggYW4gb3B0aW1pemVyLCBsb3NzLCBhbmQvb3IgbWV0cmljcy4gIENhbGxpbmcgYGZpdGBcbiAgICogb3IgYGV2YWx1YXRlYCBvbiBhbiB1bi1jb21waWxlZCBtb2RlbCB3aWxsIHRocm93IGFuIGVycm9yLlxuICAgKlxuICAgKiBAcGFyYW0gYXJncyBhIGBNb2RlbENvbXBpbGVBcmdzYCBzcGVjaWZ5aW5nIHRoZSBsb3NzLCBvcHRpbWl6ZXIsIGFuZFxuICAgKiBtZXRyaWNzIHRvIGJlIHVzZWQgZm9yIGZpdHRpbmcgYW5kIGV2YWx1YXRpbmcgdGhpcyBtb2RlbC5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGNvbXBpbGUoYXJnczogTW9kZWxDb21waWxlQXJncyk6IHZvaWQge1xuICAgIGlmIChhcmdzLmxvc3MgPT0gbnVsbCkge1xuICAgICAgYXJncy5sb3NzID0gW107XG4gICAgfVxuICAgIHRoaXMubG9zcyA9IGFyZ3MubG9zcztcblxuICAgIGlmICh0eXBlb2YgYXJncy5vcHRpbWl6ZXIgPT09ICdzdHJpbmcnKSB7XG4gICAgICB0aGlzLm9wdGltaXplcl8gPSBvcHRpbWl6ZXJzLmdldE9wdGltaXplcihhcmdzLm9wdGltaXplcik7XG4gICAgICB0aGlzLmlzT3B0aW1pemVyT3duZWQgPSB0cnVlO1xuICAgIH0gZWxzZSB7XG4gICAgICBpZiAoIShhcmdzLm9wdGltaXplciBpbnN0YW5jZW9mIE9wdGltaXplcikpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgVXNlci1kZWZpbmVkIG9wdGltaXplciBtdXN0IGJlIGFuIGluc3RhbmNlIG9mIHRmLk9wdGltaXplci5gKTtcbiAgICAgIH1cbiAgICAgIHRoaXMub3B0aW1pemVyXyA9IGFyZ3Mub3B0aW1pemVyO1xuICAgICAgdGhpcy5pc09wdGltaXplck93bmVkID0gZmFsc2U7XG4gICAgfVxuXG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIGxvc3NXZWlnaHRzLlxuICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBzYW1wbGVXZWlnaHRNb2RlLlxuXG4gICAgLy8gUHJlcGFyZSBsb3NzIGZ1bmN0aW9ucy5cbiAgICBsZXQgbG9zc0Z1bmN0aW9uczogTG9zc09yTWV0cmljRm5bXSA9IFtdO1xuICAgIGlmICghQXJyYXkuaXNBcnJheShhcmdzLmxvc3MpICYmIHR5cGVvZiBhcmdzLmxvc3MgIT09ICdzdHJpbmcnICYmXG4gICAgICAgIHR5cGVvZiBhcmdzLmxvc3MgIT09ICdmdW5jdGlvbicpIHtcbiAgICAgIGFyZ3MubG9zcyA9IGFyZ3MubG9zcyBhcyB7W291dHB1dE5hbWU6IHN0cmluZ106IHN0cmluZ307XG4gICAgICBmb3IgKGNvbnN0IG5hbWUgaW4gYXJncy5sb3NzKSB7XG4gICAgICAgIGlmICh0aGlzLm91dHB1dE5hbWVzLmluZGV4T2YobmFtZSkgPT09IC0xKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGBVbmtub3duIGVudHJ5IGluIGxvc3MgZGljdGlvbmFyeTogXCIke25hbWV9XCIuIGAgK1xuICAgICAgICAgICAgICBgT25seSBleHBlY3RlZCB0aGUgZm9sbG93aW5nIGtleXM6ICR7dGhpcy5vdXRwdXROYW1lc31gKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgZm9yIChjb25zdCBuYW1lIG9mIHRoaXMub3V0cHV0TmFtZXMpIHtcbiAgICAgICAgaWYgKGFyZ3MubG9zc1tuYW1lXSA9PSBudWxsKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgICBgT3V0cHV0IFwiJHtuYW1lfVwiIGlzIG1pc3NpbmcgZnJvbSBsb3NzIGRpY3Rpb25hcnkuIFdlIGFzc3VtZSBgICtcbiAgICAgICAgICAgICAgYHRoaXMgd2FzIGRvbmUgb24gcHVycG9zZSwgYW5kIHdlIHdpbGwgbm90IGJlIGV4cGVjdGluZyBkYXRhIGAgK1xuICAgICAgICAgICAgICBgdG8gYmUgcGFzc2VkIHRvICR7bmFtZX0gZHVyaW5nIHRyYWluaW5nYCk7XG4gICAgICAgIH1cbiAgICAgICAgbG9zc0Z1bmN0aW9ucy5wdXNoKGxvc3Nlcy5nZXQoYXJncy5sb3NzW25hbWVdKSk7XG4gICAgICB9XG4gICAgfSBlbHNlIGlmIChBcnJheS5pc0FycmF5KGFyZ3MubG9zcykpIHtcbiAgICAgIGlmIChhcmdzLmxvc3MubGVuZ3RoICE9PSB0aGlzLm91dHB1dHMubGVuZ3RoKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgYFdoZW4gcGFzc2luZyBhbiBBcnJheSBhcyBsb3NzLCBpdCBzaG91bGQgaGF2ZSBvbmUgZW50cnkgcGVyIGAgK1xuICAgICAgICAgICAgYG1vZGVsIG91dHB1dC4gVGhlIG1vZGVsIGhhcyAke3RoaXMub3V0cHV0cy5sZW5ndGh9IG91dHB1dChzKSwgYCArXG4gICAgICAgICAgICBgYnV0IHlvdSBwYXNzZWQgbG9zcz0ke2FyZ3MubG9zc30uYCk7XG4gICAgICB9XG4gICAgICBjb25zdCB0aGVMb3NzZXMgPSBhcmdzLmxvc3MgYXMgQXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPjtcbiAgICAgIGxvc3NGdW5jdGlvbnMgPSB0aGVMb3NzZXMubWFwKGwgPT4gbG9zc2VzLmdldChsKSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGxvc3NGdW5jdGlvbiA9IGxvc3Nlcy5nZXQoYXJncy5sb3NzKTtcbiAgICAgIHRoaXMub3V0cHV0cy5mb3JFYWNoKF8gPT4ge1xuICAgICAgICBsb3NzRnVuY3Rpb25zLnB1c2gobG9zc0Z1bmN0aW9uKTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIHRoaXMubG9zc0Z1bmN0aW9ucyA9IGxvc3NGdW5jdGlvbnM7XG5cbiAgICB0aGlzLmZlZWRPdXRwdXROYW1lcyA9IFtdO1xuICAgIHRoaXMuZmVlZE91dHB1dFNoYXBlcyA9IFtdO1xuICAgIHRoaXMuZmVlZExvc3NGbnMgPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMub3V0cHV0cy5sZW5ndGg7ICsraSkge1xuICAgICAgLy8gVE9ETyhjYWlzKTogTG9naWMgZm9yIHNraXBwaW5nIHRhcmdldChzKS5cbiAgICAgIGNvbnN0IHNoYXBlID0gdGhpcy5pbnRlcm5hbE91dHB1dFNoYXBlc1tpXTtcbiAgICAgIGNvbnN0IG5hbWUgPSB0aGlzLm91dHB1dE5hbWVzW2ldO1xuICAgICAgdGhpcy5mZWVkT3V0cHV0TmFtZXMucHVzaChuYW1lKTtcbiAgICAgIHRoaXMuZmVlZE91dHB1dFNoYXBlcy5wdXNoKHNoYXBlKTtcbiAgICAgIHRoaXMuZmVlZExvc3NGbnMucHVzaCh0aGlzLmxvc3NGdW5jdGlvbnNbaV0pO1xuICAgIH1cblxuICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBsb2dpYyBmb3Igb3V0cHV0IG1hc2tzLlxuICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBsb2dpYyBmb3Igc2FtcGxlIHdlaWdodHMuXG4gICAgY29uc3Qgc2tpcFRhcmdldEluZGljZXM6IG51bWJlcltdID0gW107XG5cbiAgICAvLyBQcmVwYXJlIG1ldHJpY3MuXG4gICAgdGhpcy5tZXRyaWNzID0gYXJncy5tZXRyaWNzO1xuICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCB3ZWlnaHRlZE1ldHJpY3MuXG4gICAgdGhpcy5tZXRyaWNzTmFtZXMgPSBbJ2xvc3MnXTtcbiAgICB0aGlzLm1ldHJpY3NUZW5zb3JzID0gW107XG5cbiAgICAvLyBDb21wdXRlIHRvdGFsIGxvc3MuXG4gICAgLy8gUG9ydGluZyBOb3RlOiBJbiBQeUtlcmFzLCBtZXRyaWNzX3RlbnNvcnMgYXJlIHN5bWJvbGljIHRlbnNvciBvYmplY3RzLlxuICAgIC8vICAgSGVyZSwgbWV0cmljc1RlbnNvcnMgYXJlIFR5cGVTY3JpcHQgZnVuY3Rpb25zLiBUaGlzIGRpZmZlcmVuY2UgaXMgZHVlXG4gICAgLy8gICB0byB0aGUgZGlmZmVyZW5jZSBpbiBzeW1ib2xpYy9pbXBlcmF0aXZlIHByb3BlcnR5IG9mIHRoZSBiYWNrZW5kcy5cbiAgICBuYW1lU2NvcGUoJ2xvc3MnLCAoKSA9PiB7XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMub3V0cHV0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICBpZiAoc2tpcFRhcmdldEluZGljZXMuaW5kZXhPZihpKSAhPT0gLTEpIHtcbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuICAgICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgd2VpZ2h0ZWRMb3NzLCBzYW1wbGVXZWlnaHQgYW5kIG1hc2suXG4gICAgICAgIC8vICAgVGhlIGZvbGxvd2luZyBsaW5lIHNob3VsZCBiZSB3ZWlnaHRlZExvc3NcbiAgICAgICAgY29uc3Qgd2VpZ2h0ZWRMb3NzID0gdGhpcy5sb3NzRnVuY3Rpb25zW2ldO1xuICAgICAgICBpZiAodGhpcy5vdXRwdXRzLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgICB0aGlzLm1ldHJpY3NUZW5zb3JzLnB1c2goW3dlaWdodGVkTG9zcywgaV0pO1xuICAgICAgICAgIHRoaXMubWV0cmljc05hbWVzLnB1c2godGhpcy5vdXRwdXROYW1lc1tpXSArICdfbG9zcycpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIC8vIFBvcnRpbmcgTm90ZTogRHVlIHRvIHRoZSBpbXBlcmF0aXZlIG5hdHVyZSBvZiB0aGUgYmFja2VuZCwgd2UgY2FsY3VsYXRlXG4gICAgICAvLyAgIHRoZSByZWd1bGFyaXplciBwZW5hbHRpZXMgaW4gdGhlIHRvdGFsTG9zc0Z1bmN0aW9uLCBpbnN0ZWFkIG9mIGhlcmUuXG4gICAgfSk7XG5cbiAgICBjb25zdCBuZXN0ZWRNZXRyaWNzID0gY29sbGVjdE1ldHJpY3MoYXJncy5tZXRyaWNzLCB0aGlzLm91dHB1dE5hbWVzKTtcbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgbmVzdGVkV2VpZ2h0ZWRNZXRyaWNzLlxuXG4gICAgLyoqXG4gICAgICogSGVscGVyIGZ1bmN0aW9uIHVzZWQgaW4gbG9vcCBiZWxvdy5cbiAgICAgKi9cbiAgICBjb25zdCBhcHBlbmRNZXRyaWMgPVxuICAgICAgICAob3V0cHV0SW5kZXg6IG51bWJlciwgbWV0cmljTmFtZTogc3RyaW5nLFxuICAgICAgICAgbWV0cmljVGVuc29yOiBMb3NzT3JNZXRyaWNGbikgPT4ge1xuICAgICAgICAgIGlmICh0aGlzLm91dHB1dE5hbWVzLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgICAgIG1ldHJpY05hbWUgPSB0aGlzLm91dHB1dE5hbWVzW291dHB1dEluZGV4XSArICdfJyArIG1ldHJpY05hbWU7XG4gICAgICAgICAgfVxuICAgICAgICAgIHRoaXMubWV0cmljc05hbWVzLnB1c2gobWV0cmljTmFtZSk7XG4gICAgICAgICAgdGhpcy5tZXRyaWNzVGVuc29ycy5wdXNoKFttZXRyaWNUZW5zb3IsIG91dHB1dEluZGV4XSk7XG4gICAgICAgIH07XG5cbiAgICBuYW1lU2NvcGUoJ21ldHJpYycsICgpID0+IHtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5vdXRwdXRzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgIGlmIChza2lwVGFyZ2V0SW5kaWNlcy5pbmRleE9mKGkpICE9PSAtMSkge1xuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IG91dHB1dE1ldHJpY3MgPSBuZXN0ZWRNZXRyaWNzW2ldO1xuICAgICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgd2VpZ2h0cyBhbmQgb3V0cHV0V2VpZ2h0ZWRNZXRyaWNzLlxuXG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBvcHRpb25hbCBhcmcgYHdlaWdodHNgIHRvIHRoZSBmb2xsb3dpbmcgZnVuY3Rpb24uXG4gICAgICAgIGNvbnN0IGhhbmRsZU1ldHJpY3MgPSAobWV0cmljczogQXJyYXk8c3RyaW5nfExvc3NPck1ldHJpY0ZuPikgPT4ge1xuICAgICAgICAgIGNvbnN0IG1ldHJpY05hbWVQcmVmaXggPSAnJztcbiAgICAgICAgICBsZXQgbWV0cmljTmFtZTogc3RyaW5nO1xuICAgICAgICAgIGxldCBhY2NGbjogTG9zc09yTWV0cmljRm47XG4gICAgICAgICAgbGV0IHdlaWdodGVkTWV0cmljRm46IExvc3NPck1ldHJpY0ZuO1xuICAgICAgICAgIC8vICBUT0RPKGNhaXMpOiBVc2UgJ3dlaWdodHNfJyBmb3Igd2VpZ2h0ZWQgbWV0cmljcy5cblxuICAgICAgICAgIGZvciAoY29uc3QgbWV0cmljIG9mIG1ldHJpY3MpIHtcbiAgICAgICAgICAgIGlmICh0eXBlb2YgbWV0cmljID09PSAnc3RyaW5nJyAmJlxuICAgICAgICAgICAgICAgIFsnYWNjdXJhY3knLCAnYWNjJywgJ2Nyb3NzZW50cm9weScsICdjZSddLmluZGV4T2YobWV0cmljKSAhPT1cbiAgICAgICAgICAgICAgICAgICAgLTEpIHtcbiAgICAgICAgICAgICAgY29uc3Qgb3V0cHV0U2hhcGUgPSB0aGlzLmludGVybmFsT3V0cHV0U2hhcGVzW2ldO1xuXG4gICAgICAgICAgICAgIGlmIChvdXRwdXRTaGFwZVtvdXRwdXRTaGFwZS5sZW5ndGggLSAxXSA9PT0gMSB8fFxuICAgICAgICAgICAgICAgICAgdGhpcy5sb3NzRnVuY3Rpb25zW2ldID09PSBsb3NzZXMuYmluYXJ5Q3Jvc3NlbnRyb3B5KSB7XG4gICAgICAgICAgICAgICAgLy8gY2FzZTogYmluYXJ5IGFjY3VyYWN5L2Nyb3NzZW50cm9weS5cbiAgICAgICAgICAgICAgICBpZiAoWydhY2N1cmFjeScsICdhY2MnXS5pbmRleE9mKG1ldHJpYykgIT09IC0xKSB7XG4gICAgICAgICAgICAgICAgICBhY2NGbiA9IE1ldHJpY3MuYmluYXJ5QWNjdXJhY3k7XG4gICAgICAgICAgICAgICAgfSBlbHNlIGlmIChbJ2Nyb3NzZW50cm9weScsICdjZSddLmluZGV4T2YobWV0cmljKSAhPT0gLTEpIHtcbiAgICAgICAgICAgICAgICAgIGFjY0ZuID0gTWV0cmljcy5iaW5hcnlDcm9zc2VudHJvcHk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICB9IGVsc2UgaWYgKFxuICAgICAgICAgICAgICAgICAgdGhpcy5sb3NzRnVuY3Rpb25zW2ldID09PVxuICAgICAgICAgICAgICAgICAgbG9zc2VzLnNwYXJzZUNhdGVnb3JpY2FsQ3Jvc3NlbnRyb3B5KSB7XG4gICAgICAgICAgICAgICAgLy8gY2FzZTogY2F0ZWdvcmljYWwgYWNjdXJhY3kgLyBjcm9zc2VudHJvcHkgd2l0aCBzcGFyc2VcbiAgICAgICAgICAgICAgICAvLyB0YXJnZXRzLlxuICAgICAgICAgICAgICAgIGlmIChbJ2FjY3VyYWN5JywgJ2FjYyddLmluZGV4T2YobWV0cmljKSAhPT0gLTEpIHtcbiAgICAgICAgICAgICAgICAgIGFjY0ZuID0gTWV0cmljcy5zcGFyc2VDYXRlZ29yaWNhbEFjY3VyYWN5O1xuICAgICAgICAgICAgICAgIH0gZWxzZSBpZiAoWydjcm9zc2VudHJvcHknLCAnY2UnXS5pbmRleE9mKG1ldHJpYykgIT09IC0xKSB7XG4gICAgICAgICAgICAgICAgICBhY2NGbiA9IE1ldHJpY3Muc3BhcnNlQ2F0ZWdvcmljYWxDcm9zc2VudHJvcHk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIC8vIGNhc2U6IGNhdGVnb3JpY2FsIGFjY3VyYWN5IC8gY3Jvc3NlbnRyb3B5LlxuICAgICAgICAgICAgICAgIGlmIChbJ2FjY3VyYWN5JywgJ2FjYyddLmluZGV4T2YobWV0cmljKSAhPT0gLTEpIHtcbiAgICAgICAgICAgICAgICAgIGFjY0ZuID0gTWV0cmljcy5jYXRlZ29yaWNhbEFjY3VyYWN5O1xuICAgICAgICAgICAgICAgIH0gZWxzZSBpZiAoWydjcm9zc2VudHJvcHknLCAnY2UnXS5pbmRleE9mKG1ldHJpYykgIT09IC0xKSB7XG4gICAgICAgICAgICAgICAgICBhY2NGbiA9IE1ldHJpY3MuY2F0ZWdvcmljYWxDcm9zc2VudHJvcHk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIGxldCBzdWZmaXg6IHN0cmluZztcbiAgICAgICAgICAgICAgaWYgKFsnYWNjdXJhY3knLCAnYWNjJ10uaW5kZXhPZihtZXRyaWMpICE9PSAtMSkge1xuICAgICAgICAgICAgICAgIHN1ZmZpeCA9ICdhY2MnO1xuICAgICAgICAgICAgICB9IGVsc2UgaWYgKFsnY3Jvc3NlbnRyb3B5JywgJ2NlJ10uaW5kZXhPZihtZXRyaWMpICE9PSAtMSkge1xuICAgICAgICAgICAgICAgIHN1ZmZpeCA9ICdjZSc7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgLy8gVE9ETyhjYWlzKTogQWRkIHdlaWdodGluZyBhY3R1YWxseS5cbiAgICAgICAgICAgICAgd2VpZ2h0ZWRNZXRyaWNGbiA9IGFjY0ZuO1xuICAgICAgICAgICAgICBtZXRyaWNOYW1lID0gbWV0cmljTmFtZVByZWZpeCArIHN1ZmZpeDtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIGNvbnN0IG1ldHJpY0ZuID0gTWV0cmljcy5nZXQobWV0cmljKTtcbiAgICAgICAgICAgICAgLy8gVE9ETyhjYWlzKTogQWRkIHdlaWdodGluZyBhY3R1YWxseS5cbiAgICAgICAgICAgICAgd2VpZ2h0ZWRNZXRyaWNGbiA9IG1ldHJpY0ZuO1xuICAgICAgICAgICAgICBtZXRyaWNOYW1lID1cbiAgICAgICAgICAgICAgICAgIG1ldHJpY05hbWVQcmVmaXggKyBNZXRyaWNzLmdldExvc3NPck1ldHJpY05hbWUobWV0cmljKTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgLy8gVE9ETyhjYWlzKTogQWRkIHdlaWdodGluZyBhbmQgbWFza2luZyB0byBtZXRyaWNSZXN1bHQuXG4gICAgICAgICAgICBsZXQgbWV0cmljUmVzdWx0OiBMb3NzT3JNZXRyaWNGbjtcbiAgICAgICAgICAgIG5hbWVTY29wZShtZXRyaWNOYW1lLCAoKSA9PiB7XG4gICAgICAgICAgICAgIG1ldHJpY1Jlc3VsdCA9IHdlaWdodGVkTWV0cmljRm47XG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIGFwcGVuZE1ldHJpYyhpLCBtZXRyaWNOYW1lLCBtZXRyaWNSZXN1bHQpO1xuICAgICAgICAgIH1cbiAgICAgICAgfTtcblxuICAgICAgICBoYW5kbGVNZXRyaWNzKG91dHB1dE1ldHJpY3MpO1xuICAgICAgICAvLyBUT0RPKGNhaXMpOiBDYWxsIGhhbmRsZU1ldHJpY3Mgd2l0aCB3ZWlnaHRzLlxuICAgICAgfVxuICAgIH0pO1xuXG4gICAgLy8gUG9ydGluZyBOb3RlczogR2l2ZW4gdGhlIGltcGVyYXRpdmUgYmFja2VuZCBvZiB0ZmpzLWNvcmUsXG4gICAgLy8gICB0aGVyZSBpcyBubyBuZWVkIGZvciBjb25zdHJ1Y3RpbmcgdGhlIHN5bWJvbGljIGdyYXBoIGFuZCBwbGFjZWhvbGRlcnMuXG4gICAgdGhpcy5jb2xsZWN0ZWRUcmFpbmFibGVXZWlnaHRzID0gdGhpcy50cmFpbmFibGVXZWlnaHRzO1xuICB9XG5cbiAgLyoqXG4gICAqIENoZWNrIHRyYWluYWJsZSB3ZWlnaHRzIGNvdW50IGNvbnNpc3RlbmN5LlxuICAgKlxuICAgKiBUaGlzIHdpbGwgcmFpc2UgYSB3YXJuaW5nIGlmIGB0aGlzLnRyYWluYWJsZVdlaWdodHNgIGFuZFxuICAgKiBgdGhpcy5jb2xsZWN0ZWRUcmFpbmFibGVXZWlnaHRzYCBhcmUgaW5jb25zaXN0ZW50IChpLmUuLCBoYXZlIGRpZmZlcmVudFxuICAgKiBudW1iZXJzIG9mIHBhcmFtZXRlcnMpLlxuICAgKiBJbmNvbnNpc3RlbmN5IHdpbGwgdHlwaWNhbGx5IGFyaXNlIHdoZW4gb25lIG1vZGlmaWVzIGBtb2RlbC50cmFpbmFibGVgXG4gICAqIHdpdGhvdXQgY2FsbGluZyBgbW9kZWwuY29tcGlsZSgpYCBhZ2Fpbi5cbiAgICovXG4gIHByb3RlY3RlZCBjaGVja1RyYWluYWJsZVdlaWdodHNDb25zaXN0ZW5jeSgpOiB2b2lkIHtcbiAgICBpZiAodGhpcy5jb2xsZWN0ZWRUcmFpbmFibGVXZWlnaHRzID09IG51bGwpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHRoaXMudHJhaW5hYmxlV2VpZ2h0cy5sZW5ndGggIT09XG4gICAgICAgIHRoaXMuY29sbGVjdGVkVHJhaW5hYmxlV2VpZ2h0cy5sZW5ndGgpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAnRGlzY3JlcGFuY3kgYmV0d2VlbiB0cmFpbmFibGV3ZWlnaHRzIGFuZCBjb2xsZWN0ZWQgdHJhaW5hYmxlICcgK1xuICAgICAgICAgICd3ZWlnaHRzLiBEaWQgeW91IHNldCBgbW9kZWwudHJhaW5hYmxlYCB3aXRob3V0IGNhbGxpbmcgJyArXG4gICAgICAgICAgJ2Btb2RlbC5jb21waWxlKClgIGFmdGVyd2FyZHM/Jyk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgdGhlIGxvc3MgdmFsdWUgJiBtZXRyaWNzIHZhbHVlcyBmb3IgdGhlIG1vZGVsIGluIHRlc3QgbW9kZS5cbiAgICpcbiAgICogTG9zcyBhbmQgbWV0cmljcyBhcmUgc3BlY2lmaWVkIGR1cmluZyBgY29tcGlsZSgpYCwgd2hpY2ggbmVlZHMgdG8gaGFwcGVuXG4gICAqIGJlZm9yZSBjYWxscyB0byBgZXZhbHVhdGUoKWAuXG4gICAqXG4gICAqIENvbXB1dGF0aW9uIGlzIGRvbmUgaW4gYmF0Y2hlcy5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKHtcbiAgICogICBsYXllcnM6IFt0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbMTBdfSldXG4gICAqIH0pO1xuICAgKiBtb2RlbC5jb21waWxlKHtvcHRpbWl6ZXI6ICdzZ2QnLCBsb3NzOiAnbWVhblNxdWFyZWRFcnJvcid9KTtcbiAgICogY29uc3QgcmVzdWx0ID0gbW9kZWwuZXZhbHVhdGUoXG4gICAqICAgICB0Zi5vbmVzKFs4LCAxMF0pLCB0Zi5vbmVzKFs4LCAxXSksIHtiYXRjaFNpemU6IDR9KTtcbiAgICogcmVzdWx0LnByaW50KCk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0geCBgdGYuVGVuc29yYCBvZiB0ZXN0IGRhdGEsIG9yIGFuIGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzIGlmIHRoZVxuICAgKiBtb2RlbCBoYXMgbXVsdGlwbGUgaW5wdXRzLlxuICAgKiBAcGFyYW0geSBgdGYuVGVuc29yYCBvZiB0YXJnZXQgZGF0YSwgb3IgYW4gYEFycmF5YCBvZiBgdGYuVGVuc29yYHMgaWYgdGhlXG4gICAqIG1vZGVsIGhhcyBtdWx0aXBsZSBvdXRwdXRzLlxuICAgKiBAcGFyYW0gYXJncyBBIGBNb2RlbEV2YWx1YXRlQXJnc2AsIGNvbnRhaW5pbmcgb3B0aW9uYWwgZmllbGRzLlxuICAgKlxuICAgKiBAcmV0dXJuIGBTY2FsYXJgIHRlc3QgbG9zcyAoaWYgdGhlIG1vZGVsIGhhcyBhIHNpbmdsZSBvdXRwdXQgYW5kIG5vXG4gICAqICAgbWV0cmljcykgb3IgYEFycmF5YCBvZiBgU2NhbGFyYHMgKGlmIHRoZSBtb2RlbCBoYXMgbXVsdGlwbGUgb3V0cHV0c1xuICAgKiAgIGFuZC9vciBtZXRyaWNzKS4gVGhlIGF0dHJpYnV0ZSBgbW9kZWwubWV0cmljc05hbWVzYFxuICAgKiAgIHdpbGwgZ2l2ZSB5b3UgdGhlIGRpc3BsYXkgbGFiZWxzIGZvciB0aGUgc2NhbGFyIG91dHB1dHMuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBldmFsdWF0ZShcbiAgICAgIHg6IFRlbnNvcnxUZW5zb3JbXSwgeTogVGVuc29yfFRlbnNvcltdLFxuICAgICAgYXJnczogTW9kZWxFdmFsdWF0ZUFyZ3MgPSB7fSk6IFNjYWxhcnxTY2FsYXJbXSB7XG4gICAgY29uc3QgYmF0Y2hTaXplID0gYXJncy5iYXRjaFNpemUgPT0gbnVsbCA/IDMyIDogYXJncy5iYXRjaFNpemU7XG4gICAgY2hlY2tCYXRjaFNpemUoYmF0Y2hTaXplKTtcblxuICAgIC8vIFRPRE8oY2Fpcyk6IFN0YW5kYXJkaXplIGBjb25maWcuc2FtcGxlV2VpZ2h0c2AgYXMgd2VsbC5cbiAgICAvLyBWYWxpZGF0ZSB1c2VyIGRhdGEuXG4gICAgY29uc3QgY2hlY2tCYXRjaEF4aXMgPSB0cnVlO1xuICAgIGNvbnN0IHN0YW5kYXJkaXplZE91dHMgPVxuICAgICAgICB0aGlzLnN0YW5kYXJkaXplVXNlckRhdGFYWSh4LCB5LCBjaGVja0JhdGNoQXhpcywgYmF0Y2hTaXplKTtcbiAgICB0cnkge1xuICAgICAgLy8gVE9ETyhjYWlzKTogSWYgdXNlcyBgdXNlTGVhcm5pbmdQaGFzZWAsIHNldCB0aGUgY29ycmVzcG9uZGluZyBlbGVtZW50XG4gICAgICAvLyBvZiB0aGUgaW5wdXQgdG8gMC5cbiAgICAgIGNvbnN0IGlucyA9IHN0YW5kYXJkaXplZE91dHNbMF0uY29uY2F0KHN0YW5kYXJkaXplZE91dHNbMV0pO1xuICAgICAgdGhpcy5tYWtlVGVzdEZ1bmN0aW9uKCk7XG4gICAgICBjb25zdCBmID0gdGhpcy50ZXN0RnVuY3Rpb247XG4gICAgICBjb25zdCB0ZXN0T3V0cyA9XG4gICAgICAgICAgdGhpcy50ZXN0TG9vcChmLCBpbnMsIGJhdGNoU2l6ZSwgYXJncy52ZXJib3NlLCBhcmdzLnN0ZXBzKTtcbiAgICAgIHJldHVybiBzaW5nbGV0b25PckFycmF5KHRlc3RPdXRzKTtcbiAgICB9IGZpbmFsbHkge1xuICAgICAgZGlzcG9zZU5ld1RlbnNvcnMoc3RhbmRhcmRpemVkT3V0c1swXSwgeCk7XG4gICAgICBkaXNwb3NlTmV3VGVuc29ycyhzdGFuZGFyZGl6ZWRPdXRzWzFdLCB5KTtcbiAgICB9XG4gIH1cblxuICAvLyBUT0RPKGNhaXMpOiBBZGQgY29kZSBzbmlwcGV0IGJlbG93IG9uY2UgcmVhbCBkYXRhc2V0IG9iamVjdHMgYXJlXG4gIC8vICAgYXZhaWxhYmxlLlxuICAvKipcbiAgICogRXZhbHVhdGUgbW9kZWwgdXNpbmcgYSBkYXRhc2V0IG9iamVjdC5cbiAgICpcbiAgICogTm90ZTogVW5saWtlIGBldmFsdWF0ZSgpYCwgdGhpcyBtZXRob2QgaXMgYXN5bmNocm9ub3VzIChgYXN5bmNgKS5cbiAgICpcbiAgICogQHBhcmFtIGRhdGFzZXQgQSBkYXRhc2V0IG9iamVjdC4gSXRzIGBpdGVyYXRvcigpYCBtZXRob2QgaXMgZXhwZWN0ZWRcbiAgICogICB0byBnZW5lcmF0ZSBhIGRhdGFzZXQgaXRlcmF0b3Igb2JqZWN0LCB0aGUgYG5leHQoKWAgbWV0aG9kIG9mIHdoaWNoXG4gICAqICAgaXMgZXhwZWN0ZWQgdG8gcHJvZHVjZSBkYXRhIGJhdGNoZXMgZm9yIGV2YWx1YXRpb24uIFRoZSByZXR1cm4gdmFsdWVcbiAgICogICBvZiB0aGUgYG5leHQoKWAgY2FsbCBvdWdodCB0byBjb250YWluIGEgYm9vbGVhbiBgZG9uZWAgZmllbGQgYW5kIGFcbiAgICogICBgdmFsdWVgIGZpZWxkLiBUaGUgYHZhbHVlYCBmaWVsZCBpcyBleHBlY3RlZCB0byBiZSBhbiBhcnJheSBvZiB0d29cbiAgICogICBgdGYuVGVuc29yYHMgb3IgYW4gYXJyYXkgb2YgdHdvIG5lc3RlZCBgdGYuVGVuc29yYCBzdHJ1Y3R1cmVzLiBUaGUgZm9ybWVyXG4gICAqICAgY2FzZSBpcyBmb3IgbW9kZWxzIHdpdGggZXhhY3RseSBvbmUgaW5wdXQgYW5kIG9uZSBvdXRwdXQgKGUuZy5cbiAgICogICBhIHNlcXVlbnRpYWwgbW9kZWwpLiBUaGUgbGF0dGVyIGNhc2UgaXMgZm9yIG1vZGVscyB3aXRoIG11bHRpcGxlXG4gICAqICAgaW5wdXRzIGFuZC9vciBtdWx0aXBsZSBvdXRwdXRzLiBPZiB0aGUgdHdvIGl0ZW1zIGluIHRoZSBhcnJheSwgdGhlXG4gICAqICAgZmlyc3QgaXMgdGhlIGlucHV0IGZlYXR1cmUocykgYW5kIHRoZSBzZWNvbmQgaXMgdGhlIG91dHB1dCB0YXJnZXQocykuXG4gICAqIEBwYXJhbSBhcmdzIEEgY29uZmlndXJhdGlvbiBvYmplY3QgZm9yIHRoZSBkYXRhc2V0LWJhc2VkIGV2YWx1YXRpb24uXG4gICAqIEByZXR1cm5zIExvc3MgYW5kIG1ldHJpYyB2YWx1ZXMgYXMgYW4gQXJyYXkgb2YgYFNjYWxhcmAgb2JqZWN0cy5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFzeW5jIGV2YWx1YXRlRGF0YXNldChkYXRhc2V0OiBEYXRhc2V0PHt9PiwgYXJncz86IE1vZGVsRXZhbHVhdGVEYXRhc2V0QXJncyk6XG4gICAgICBQcm9taXNlPFNjYWxhcnxTY2FsYXJbXT4ge1xuICAgIHRoaXMubWFrZVRlc3RGdW5jdGlvbigpO1xuICAgIHJldHVybiBldmFsdWF0ZURhdGFzZXQodGhpcywgZGF0YXNldCwgYXJncyk7XG4gIH1cblxuICAvKipcbiAgICogR2V0IG51bWJlciBvZiBzYW1wbGVzIHByb3ZpZGVkIGZvciB0cmFpbmluZywgZXZhbHVhdGlvbiBvciBwcmVkaWN0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0gaW5zIElucHV0IGB0Zi5UZW5zb3JgLlxuICAgKiBAcGFyYW0gYmF0Y2hTaXplIEludGVnZXIgYmF0Y2ggc2l6ZSwgb3B0aW9uYWwuXG4gICAqIEBwYXJhbSBzdGVwcyBUb3RhbCBudW1iZXIgb2Ygc3RlcHMgKGJhdGNoZXMgb2Ygc2FtcGxlcykgYmVmb3JlXG4gICAqIGRlY2xhcmluZyBsb29wIGZpbmlzaGVkLiBPcHRpb25hbC5cbiAgICogQHBhcmFtIHN0ZXBzTmFtZSBUaGUgcHVibGljIEFQSSdzIHBhcmFtZXRlciBuYW1lIGZvciBgc3RlcHNgLlxuICAgKiBAcmV0dXJucyBOdW1iZXIgb2Ygc2FtcGxlcyBwcm92aWRlZC5cbiAgICovXG4gIHByaXZhdGUgY2hlY2tOdW1TYW1wbGVzKFxuICAgICAgaW5zOiBUZW5zb3J8VGVuc29yW10sIGJhdGNoU2l6ZT86IG51bWJlciwgc3RlcHM/OiBudW1iZXIsXG4gICAgICBzdGVwc05hbWUgPSAnc3RlcHMnKTogbnVtYmVyIHtcbiAgICBsZXQgbnVtU2FtcGxlczogbnVtYmVyO1xuICAgIGlmIChzdGVwcyAhPSBudWxsKSB7XG4gICAgICBudW1TYW1wbGVzID0gbnVsbDtcbiAgICAgIGlmIChiYXRjaFNpemUgIT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBJZiAke3N0ZXBzTmFtZX0gaXMgc2V0LCBiYXRjaFNpemUgbXVzdCBiZSBudWxsIG9yIHVuZGVmaW5lZC5gICtcbiAgICAgICAgICAgIGBHb3QgYmF0Y2hTaXplID0gJHtiYXRjaFNpemV9YCk7XG4gICAgICB9XG4gICAgfSBlbHNlIGlmIChpbnMgIT0gbnVsbCkge1xuICAgICAgaWYgKEFycmF5LmlzQXJyYXkoaW5zKSkge1xuICAgICAgICBudW1TYW1wbGVzID0gaW5zWzBdLnNoYXBlWzBdO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgbnVtU2FtcGxlcyA9IGlucy5zaGFwZVswXTtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYEVpdGhlciB0aGUgaW5wdXQgZGF0YSBzaG91bGQgaGF2ZSBhIGRlZmluZWQgc2hhcGUsIG9yIGAgK1xuICAgICAgICAgIGAke3N0ZXBzTmFtZX0gc2hvdWQgYmUgc3BlY2lmaWVkLmApO1xuICAgIH1cbiAgICByZXR1cm4gbnVtU2FtcGxlcztcbiAgfVxuXG4gIC8qKlxuICAgKiBFeGVjdXRlIGludGVybmFsIHRlbnNvcnMgb2YgdGhlIG1vZGVsIHdpdGggaW5wdXQgZGF0YSBmZWVkLlxuICAgKiBAcGFyYW0gaW5wdXRzIElucHV0IGRhdGEgZmVlZC4gTXVzdCBtYXRjaCB0aGUgaW5wdXRzIG9mIHRoZSBtb2RlbC5cbiAgICogQHBhcmFtIG91dHB1dHMgTmFtZXMgb2YgdGhlIG91dHB1dCB0ZW5zb3JzIHRvIGJlIGZldGNoZWQuIE11c3QgbWF0Y2hcbiAgICogICBuYW1lcyBvZiB0aGUgU3ltYm9saWNUZW5zb3JzIHRoYXQgYmVsb25nIHRvIHRoZSBncmFwaC5cbiAgICogQHJldHVybnMgRmV0Y2hlZCB2YWx1ZXMgZm9yIGBvdXRwdXRzYC5cbiAgICovXG4gIGV4ZWN1dGUoaW5wdXRzOiBUZW5zb3J8VGVuc29yW118TmFtZWRUZW5zb3JNYXAsIG91dHB1dHM6IHN0cmluZ3xzdHJpbmdbXSk6XG4gICAgICBUZW5zb3J8VGVuc29yW10ge1xuICAgIGlmIChBcnJheS5pc0FycmF5KG91dHB1dHMpICYmIG91dHB1dHMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAnYG91dHB1dHNgIGlzIGFuIGVtcHR5IEFycmF5LCB3aGljaCBpcyBub3QgYWxsb3dlZC4nKTtcbiAgICB9XG5cbiAgICBjb25zdCBvdXRwdXRzSXNBcnJheSA9IEFycmF5LmlzQXJyYXkob3V0cHV0cyk7XG4gICAgY29uc3Qgb3V0cHV0TmFtZXMgPVxuICAgICAgICAob3V0cHV0c0lzQXJyYXkgPyBvdXRwdXRzIDogW291dHB1dHNdKTtcbiAgICBjb25zdCBvdXRwdXRTeW1ib2xpY1RlbnNvcnMgPSB0aGlzLnJldHJpZXZlU3ltYm9saWNUZW5zb3JzKG91dHB1dE5hbWVzKTtcblxuICAgIC8vIEZvcm1hdCB0aGUgaW5wdXQgaW50byBhIEZlZWREaWN0LlxuICAgIGNvbnN0IGZlZWREaWN0ID0gbmV3IEZlZWREaWN0KCk7XG4gICAgaWYgKGlucHV0cyBpbnN0YW5jZW9mIFRlbnNvcikge1xuICAgICAgaW5wdXRzID0gW2lucHV0c107XG4gICAgfVxuICAgIGlmIChBcnJheS5pc0FycmF5KGlucHV0cykpIHtcbiAgICAgIGlmIChpbnB1dHMubGVuZ3RoICE9PSB0aGlzLmlucHV0cy5sZW5ndGgpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgVGhlIG51bWJlciBvZiBpbnB1dHMgcHJvdmlkZWQgKCR7aW5wdXRzLmxlbmd0aH0pIGAgK1xuICAgICAgICAgICAgYGRvZXMgbm90IG1hdGNoIHRoZSBudW1iZXIgb2YgaW5wdXRzIG9mIHRoaXMgbW9kZWwgYCArXG4gICAgICAgICAgICBgKCR7dGhpcy5pbnB1dHMubGVuZ3RofSkuYCk7XG4gICAgICB9XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuaW5wdXRzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgIGZlZWREaWN0LmFkZCh0aGlzLmlucHV0c1tpXSwgaW5wdXRzW2ldKTtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgZm9yIChjb25zdCBpbnB1dCBvZiB0aGlzLmlucHV0cykge1xuICAgICAgICBjb25zdCB0ZW5zb3JWYWx1ZSA9IGlucHV0c1tpbnB1dC5uYW1lXTtcbiAgICAgICAgaWYgKHRlbnNvclZhbHVlID09IG51bGwpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICAgYE5vIHZhbHVlIGlzIHByb3ZpZGVkIGZvciB0aGUgbW9kZWwncyBpbnB1dCAke2lucHV0Lm5hbWV9YCk7XG4gICAgICAgIH1cbiAgICAgICAgZmVlZERpY3QuYWRkKGlucHV0LCB0ZW5zb3JWYWx1ZSk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gUnVuIGV4ZWN1dGlvbi5cbiAgICBjb25zdCBleGVjdXRlT3V0cHV0cyA9IGV4ZWN1dGUob3V0cHV0U3ltYm9saWNUZW5zb3JzLCBmZWVkRGljdCkgYXMgVGVuc29yW107XG4gICAgcmV0dXJuIG91dHB1dHNJc0FycmF5ID8gZXhlY3V0ZU91dHB1dHMgOiBleGVjdXRlT3V0cHV0c1swXTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXRyaWV2ZSB0aGUgbW9kZWwncyBpbnRlcm5hbCBzeW1ib2xpYyB0ZW5zb3JzIGZyb20gc3ltYm9saWMtdGVuc29yIG5hbWVzLlxuICAgKi9cbiAgcHJpdmF0ZSByZXRyaWV2ZVN5bWJvbGljVGVuc29ycyhzeW1ib2xpY1RlbnNvck5hbWVzOiBzdHJpbmdbXSk6XG4gICAgICBTeW1ib2xpY1RlbnNvcltdIHtcbiAgICBjb25zdCBvdXRwdXRTeW1ib2xpY1RlbnNvcnM6IFN5bWJvbGljVGVuc29yW10gPVxuICAgICAgICBweUxpc3RSZXBlYXQobnVsbCwgc3ltYm9saWNUZW5zb3JOYW1lcy5sZW5ndGgpO1xuICAgIGxldCBvdXRwdXRzUmVtYWluaW5nID0gc3ltYm9saWNUZW5zb3JOYW1lcy5sZW5ndGg7XG4gICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLmxheWVycykge1xuICAgICAgY29uc3QgbGF5ZXJPdXRwdXRzOiBTeW1ib2xpY1RlbnNvcltdID1cbiAgICAgICAgICBBcnJheS5pc0FycmF5KGxheWVyLm91dHB1dCkgPyBsYXllci5vdXRwdXQgOiBbbGF5ZXIub3V0cHV0XTtcbiAgICAgIGNvbnN0IGxheWVyT3V0cHV0TmFtZXMgPSBsYXllck91dHB1dHMubWFwKG91dHB1dCA9PiBvdXRwdXQubmFtZSk7XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHN5bWJvbGljVGVuc29yTmFtZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgY29uc3QgaW5kZXggPSBsYXllck91dHB1dE5hbWVzLmluZGV4T2Yoc3ltYm9saWNUZW5zb3JOYW1lc1tpXSk7XG4gICAgICAgIGlmIChpbmRleCAhPT0gLTEpIHtcbiAgICAgICAgICBvdXRwdXRTeW1ib2xpY1RlbnNvcnNbaV0gPSBsYXllck91dHB1dHNbaW5kZXhdO1xuICAgICAgICAgIG91dHB1dHNSZW1haW5pbmctLTtcbiAgICAgICAgfVxuICAgICAgICBpZiAob3V0cHV0c1JlbWFpbmluZyA9PT0gMCkge1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBpZiAob3V0cHV0c1JlbWFpbmluZyA9PT0gMCkge1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICB9XG5cbiAgICBpZiAob3V0cHV0c1JlbWFpbmluZyA+IDApIHtcbiAgICAgIGNvbnN0IHJlbWFpbmluZ05hbWVzOiBzdHJpbmdbXSA9IFtdO1xuICAgICAgb3V0cHV0U3ltYm9saWNUZW5zb3JzLmZvckVhY2goKHRlbnNvciwgaSkgPT4ge1xuICAgICAgICBpZiAodGVuc29yID09IG51bGwpIHtcbiAgICAgICAgICByZW1haW5pbmdOYW1lcy5wdXNoKHN5bWJvbGljVGVuc29yTmFtZXNbaV0pO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBDYW5ub3QgZmluZCBTeW1ib2xpY1RlbnNvcnMgZm9yIG91dHB1dCBuYW1lKHMpOiBgICtcbiAgICAgICAgICBgJHtKU09OLnN0cmluZ2lmeShyZW1haW5pbmdOYW1lcyl9YCk7XG4gICAgfVxuICAgIHJldHVybiBvdXRwdXRTeW1ib2xpY1RlbnNvcnM7XG4gIH1cblxuICAvKipcbiAgICogSGVscGVyIG1ldGhvZCB0byBsb29wIG92ZXIgc29tZSBkYXRhIGluIGJhdGNoZXMuXG4gICAqXG4gICAqIFBvcnRpbmcgTm90ZTogTm90IHVzaW5nIHRoZSBmdW5jdGlvbmFsIGFwcHJvYWNoIGluIHRoZSBQeXRob24gZXF1aXZhbGVudFxuICAgKiAgIGR1ZSB0byB0aGUgaW1wZXJhdGl2ZSBiYWNrZW5kLlxuICAgKiBQb3J0aW5nIE5vdGU6IERvZXMgbm90IHN1cHBvcnQgc3RlcCBtb2RlIGN1cnJlbnRseS5cbiAgICpcbiAgICogQHBhcmFtIGluczogaW5wdXQgZGF0YVxuICAgKiBAcGFyYW0gYmF0Y2hTaXplOiBpbnRlZ2VyIGJhdGNoIHNpemUuXG4gICAqIEBwYXJhbSB2ZXJib3NlOiB2ZXJib3NpdHkgbW9kZWxcbiAgICogQHJldHVybnM6IFByZWRpY3Rpb25zIGFzIGB0Zi5UZW5zb3JgIChpZiBhIHNpbmdsZSBvdXRwdXQpIG9yIGFuIGBBcnJheWAgb2ZcbiAgICogICBgdGYuVGVuc29yYCAoaWYgbXVsdGlwZSBvdXRwdXRzKS5cbiAgICovXG4gIHByaXZhdGUgcHJlZGljdExvb3AoaW5zOiBUZW5zb3J8VGVuc29yW10sIGJhdGNoU2l6ZSA9IDMyLCB2ZXJib3NlID0gZmFsc2UpOlxuICAgICAgVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGZjLnRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgbnVtU2FtcGxlcyA9IHRoaXMuY2hlY2tOdW1TYW1wbGVzKGlucyk7XG4gICAgICBpZiAodmVyYm9zZSkge1xuICAgICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAgICdWZXJib3NlIHByZWRpY3RMb29wKCkgaXMgbm90IGltcGxlbWVudGVkIHlldC4nKTtcbiAgICAgIH1cblxuICAgICAgLy8gU2FtcGxlLWJhc2VkIHByZWRpY3Rpb25zLlxuICAgICAgLy8gUG9ydGluZyBOb3RlOiBUZW5zb3IgY3VycmVudGx5IGRvZXMgbm90IHN1cHBvcnQgc2xpY2VkIGFzc2lnbm1lbnRzIGFzXG4gICAgICAvLyAgIGluIG51bXB5LCBlLmcuLCB4WzE6M10gPSB5LiBUaGVyZWZvcmUgd2UgdXNlIGNvbmNhdGVuYXRpb24gd2hpbGVcbiAgICAgIC8vICAgaXRlcmF0aW5nIG92ZXIgdGhlIGJhdGNoZXMuXG5cbiAgICAgIGNvbnN0IGJhdGNoZXMgPSBtYWtlQmF0Y2hlcyhudW1TYW1wbGVzLCBiYXRjaFNpemUpO1xuICAgICAgY29uc3Qgb3V0c0JhdGNoZXM6IFRlbnNvcltdW10gPSB0aGlzLm91dHB1dHMubWFwKG91dHB1dCA9PiBbXSk7XG5cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IENhbiB0aGUgc2NvcGUoKSBiZSBwdXNoZWQgZG93biBpbnNpZGUgdGhlIGZvciBsb29wP1xuICAgICAgZm9yIChsZXQgYmF0Y2hJbmRleCA9IDA7IGJhdGNoSW5kZXggPCBiYXRjaGVzLmxlbmd0aDsgKytiYXRjaEluZGV4KSB7XG4gICAgICAgIGNvbnN0IGJhdGNoT3V0cyA9IHRmYy50aWR5KCgpID0+IHtcbiAgICAgICAgICBjb25zdCBiYXRjaFN0YXJ0ID0gYmF0Y2hlc1tiYXRjaEluZGV4XVswXTtcbiAgICAgICAgICBjb25zdCBiYXRjaEVuZCA9IGJhdGNoZXNbYmF0Y2hJbmRleF1bMV07XG4gICAgICAgICAgLy8gVE9ETyhjYWlzKTogVGFrZSBjYXJlIG9mIHRoZSBjYXNlIG9mIHRoZSBsYXN0IGVsZW1lbnQgaXMgYSBmbGFnIGZvclxuICAgICAgICAgIC8vICAgdHJhaW5pbmcvdGVzdC5cbiAgICAgICAgICBjb25zdCBpbnNCYXRjaCA9IHNsaWNlQXJyYXlzKGlucywgYmF0Y2hTdGFydCwgYmF0Y2hFbmQpO1xuXG4gICAgICAgICAgLy8gQ29uc3RydWN0IHRoZSBmZWVkcyBmb3IgZXhlY3V0ZSgpO1xuICAgICAgICAgIGNvbnN0IGZlZWRzID0gW107XG4gICAgICAgICAgaWYgKEFycmF5LmlzQXJyYXkoaW5zQmF0Y2gpKSB7XG4gICAgICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IGluc0JhdGNoLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgICAgIGZlZWRzLnB1c2goe2tleTogdGhpcy5pbnB1dHNbaV0sIHZhbHVlOiBpbnNCYXRjaFtpXX0pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBmZWVkcy5wdXNoKHtrZXk6IHRoaXMuaW5wdXRzWzBdLCB2YWx1ZTogaW5zQmF0Y2h9KTtcbiAgICAgICAgICB9XG4gICAgICAgICAgY29uc3QgZmVlZERpY3QgPSBuZXcgRmVlZERpY3QoZmVlZHMpO1xuICAgICAgICAgIHJldHVybiBleGVjdXRlKHRoaXMub3V0cHV0cywgZmVlZERpY3QpIGFzIFRlbnNvcltdO1xuICAgICAgICB9KTtcbiAgICAgICAgYmF0Y2hPdXRzLmZvckVhY2goKGJhdGNoT3V0LCBpKSA9PiBvdXRzQmF0Y2hlc1tpXS5wdXNoKGJhdGNoT3V0KSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gc2luZ2xldG9uT3JBcnJheShcbiAgICAgICAgICBvdXRzQmF0Y2hlcy5tYXAoYmF0Y2hlcyA9PiB0ZmMuY29uY2F0KGJhdGNoZXMsIDApKSk7XG4gICAgfSk7XG4gIH1cblxuICAvKipcbiAgICogR2VuZXJhdGVzIG91dHB1dCBwcmVkaWN0aW9ucyBmb3IgdGhlIGlucHV0IHNhbXBsZXMuXG4gICAqXG4gICAqIENvbXB1dGF0aW9uIGlzIGRvbmUgaW4gYmF0Y2hlcy5cbiAgICpcbiAgICogTm90ZTogdGhlIFwic3RlcFwiIG1vZGUgb2YgcHJlZGljdCgpIGlzIGN1cnJlbnRseSBub3Qgc3VwcG9ydGVkLlxuICAgKiAgIFRoaXMgaXMgYmVjYXVzZSB0aGUgVGVuc29yRmxvdy5qcyBjb3JlIGJhY2tlbmQgaXMgaW1wZXJhdGl2ZSBvbmx5LlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoe1xuICAgKiAgIGxheWVyczogW3RmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFsxMF19KV1cbiAgICogfSk7XG4gICAqIG1vZGVsLnByZWRpY3QodGYub25lcyhbOCwgMTBdKSwge2JhdGNoU2l6ZTogNH0pLnByaW50KCk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgZGF0YSwgYXMgYSBUZW5zb3IsIG9yIGFuIGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzIGlmXG4gICAqICAgdGhlIG1vZGVsIGhhcyBtdWx0aXBsZSBpbnB1dHMuXG4gICAqIEBwYXJhbSBhcmdzIEEgYE1vZGVsUHJlZGljdEFyZ3NgIG9iamVjdCBjb250YWluaW5nIG9wdGlvbmFsIGZpZWxkcy5cbiAgICpcbiAgICogQHJldHVybiBQcmVkaWN0aW9uIHJlc3VsdHMgYXMgYSBgdGYuVGVuc29yYChzKS5cbiAgICpcbiAgICogQGV4Y2VwdGlvbiBWYWx1ZUVycm9yIEluIGNhc2Ugb2YgbWlzbWF0Y2ggYmV0d2VlbiB0aGUgcHJvdmlkZWQgaW5wdXQgZGF0YVxuICAgKiAgIGFuZCB0aGUgbW9kZWwncyBleHBlY3RhdGlvbnMsIG9yIGluIGNhc2UgYSBzdGF0ZWZ1bCBtb2RlbCByZWNlaXZlcyBhXG4gICAqICAgbnVtYmVyIG9mIHNhbXBsZXMgdGhhdCBpcyBub3QgYSBtdWx0aXBsZSBvZiB0aGUgYmF0Y2ggc2l6ZS5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHByZWRpY3QoeDogVGVuc29yfFRlbnNvcltdLCBhcmdzOiBNb2RlbFByZWRpY3RBcmdzID0ge30pOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIGNvbnN0IHhzUmFuazJPckhpZ2hlciA9IGVuc3VyZVRlbnNvcnNSYW5rMk9ySGlnaGVyKHgpO1xuICAgIGNoZWNrSW5wdXREYXRhKFxuICAgICAgICB4c1JhbmsyT3JIaWdoZXIsIHRoaXMuaW5wdXROYW1lcywgdGhpcy5mZWVkSW5wdXRTaGFwZXMsIGZhbHNlKTtcbiAgICB0cnkge1xuICAgICAgLy8gVE9ETyhjYWlzKTogVGFrZSBjYXJlIG9mIHN0YXRlZnVsIG1vZGVscy5cbiAgICAgIC8vICAgaWYgKHRoaXMuc3RhdGVmdWwpIC4uLlxuICAgICAgLy8gVE9ETyhjYWlzKTogVGFrZSBjYXJlIG9mIHRoZSBsZWFybmluZ19waGFzZSBib29sZWFuIGZsYWcuXG4gICAgICAvLyAgIGlmICh0aGlzLnVzZUxlYXJuaW5nUGhhc2UpIC4uLlxuICAgICAgY29uc3QgYmF0Y2hTaXplID0gYXJncy5iYXRjaFNpemUgPT0gbnVsbCA/IDMyIDogYXJncy5iYXRjaFNpemU7XG4gICAgICBjaGVja0JhdGNoU2l6ZShiYXRjaFNpemUpO1xuICAgICAgcmV0dXJuIHRoaXMucHJlZGljdExvb3AoeHNSYW5rMk9ySGlnaGVyLCBiYXRjaFNpemUpO1xuICAgIH0gZmluYWxseSB7XG4gICAgICBkaXNwb3NlTmV3VGVuc29ycyh4c1JhbmsyT3JIaWdoZXIsIHgpO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBSZXR1cm5zIHByZWRpY3Rpb25zIGZvciBhIHNpbmdsZSBiYXRjaCBvZiBzYW1wbGVzLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoe1xuICAgKiAgIGxheWVyczogW3RmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFsxMF19KV1cbiAgICogfSk7XG4gICAqIG1vZGVsLnByZWRpY3RPbkJhdGNoKHRmLm9uZXMoWzgsIDEwXSkpLnByaW50KCk7XG4gICAqIGBgYFxuICAgKiBAcGFyYW0geDogSW5wdXQgc2FtcGxlcywgYXMgYSBUZW5zb3IgKGZvciBtb2RlbHMgd2l0aCBleGFjdGx5IG9uZVxuICAgKiAgIGlucHV0KSBvciBhbiBhcnJheSBvZiBUZW5zb3JzIChmb3IgbW9kZWxzIHdpdGggbW9yZSB0aGFuIG9uZSBpbnB1dCkuXG4gICAqIEByZXR1cm4gVGVuc29yKHMpIG9mIHByZWRpY3Rpb25zXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBwcmVkaWN0T25CYXRjaCh4OiBUZW5zb3J8VGVuc29yW10pOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIGNoZWNrSW5wdXREYXRhKHgsIHRoaXMuaW5wdXROYW1lcywgdGhpcy5mZWVkSW5wdXRTaGFwZXMsIHRydWUpO1xuICAgIC8vIFRPRE8oY2Fpcyk6IFRha2UgY2FyZSBvZiB0aGUgbGVhcm5pbmdfcGhhc2UgYm9vbGVhbiBmbGFnLlxuICAgIC8vICAgaWYgKHRoaXMudXNlTGVhcm5pbmdQaGFzZSkgLi4uXG4gICAgY29uc3QgYmF0Y2hTaXplID0gKEFycmF5LmlzQXJyYXkoeCkgPyB4WzBdIDogeCkuc2hhcGVbMF07XG4gICAgcmV0dXJuIHRoaXMucHJlZGljdExvb3AoeCwgYmF0Y2hTaXplKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzdGFuZGFyZGl6ZVVzZXJEYXRhWFkoXG4gICAgICB4OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgICB5OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sIGNoZWNrQmF0Y2hBeGlzID0gdHJ1ZSxcbiAgICAgIGJhdGNoU2l6ZT86IG51bWJlcik6IFtUZW5zb3JbXSwgVGVuc29yW11dIHtcbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgc2FtcGxlV2VpZ2h0LCBjbGFzc1dlaWdodFxuICAgIGlmICh0aGlzLm9wdGltaXplcl8gPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFJ1bnRpbWVFcnJvcihcbiAgICAgICAgICAnWW91IG11c3QgY29tcGlsZSBhIG1vZGVsIGJlZm9yZSB0cmFpbmluZy90ZXN0aW5nLiBVc2UgJyArXG4gICAgICAgICAgJ0xheWVyc01vZGVsLmNvbXBpbGUobW9kZWxDb21waWxlQXJncykuJyk7XG4gICAgfVxuICAgIGNvbnN0IG91dHB1dFNoYXBlczogU2hhcGVbXSA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5mZWVkT3V0cHV0U2hhcGVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCBvdXRwdXRTaGFwZSA9IHRoaXMuZmVlZE91dHB1dFNoYXBlc1tpXTtcbiAgICAgIGNvbnN0IGxvc3NGbiA9IHRoaXMuZmVlZExvc3NGbnNbaV07XG4gICAgICBpZiAobG9zc0ZuID09PSBsb3NzZXMuc3BhcnNlQ2F0ZWdvcmljYWxDcm9zc2VudHJvcHkpIHtcbiAgICAgICAgb3V0cHV0U2hhcGVzLnB1c2goXG4gICAgICAgICAgICBvdXRwdXRTaGFwZS5zbGljZSgwLCBvdXRwdXRTaGFwZS5sZW5ndGggLSAxKS5jb25jYXQoWzFdKSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICAvLyBQb3J0aW5nIE5vdGU6IEJlY2F1c2Ugb2Ygc3Ryb25nIHR5cGluZyBgbG9zc0ZuYCBtdXN0IGJlIGEgZnVuY3Rpb24uXG4gICAgICAgIG91dHB1dFNoYXBlcy5wdXNoKG91dHB1dFNoYXBlKTtcbiAgICAgIH1cbiAgICB9XG4gICAgeCA9IHN0YW5kYXJkaXplSW5wdXREYXRhKFxuICAgICAgICB4LCB0aGlzLmZlZWRJbnB1dE5hbWVzLCB0aGlzLmZlZWRJbnB1dFNoYXBlcywgZmFsc2UsICdpbnB1dCcpO1xuICAgIHkgPSBzdGFuZGFyZGl6ZUlucHV0RGF0YShcbiAgICAgICAgeSwgdGhpcy5mZWVkT3V0cHV0TmFtZXMsIG91dHB1dFNoYXBlcywgZmFsc2UsICd0YXJnZXQnKTtcbiAgICAvLyBUT0RPKGNhaXMpOiBTdGFuZGFyZGl6ZSBzYW1wbGVXZWlnaHRzICYgY2xhc3NXZWlnaHRzLlxuICAgIGNoZWNrQXJyYXlMZW5ndGhzKHgsIHksIG51bGwpO1xuICAgIC8vIFRPRE8oY2Fpcyk6IENoZWNrIHNhbXBsZVdlaWdodHMgYXMgd2VsbC5cbiAgICBjaGVja0xvc3NBbmRUYXJnZXRDb21wYXRpYmlsaXR5KHksIHRoaXMuZmVlZExvc3NGbnMsIHRoaXMuZmVlZE91dHB1dFNoYXBlcyk7XG4gICAgaWYgKHRoaXMuc3RhdGVmdWwgJiYgYmF0Y2hTaXplICE9IG51bGwgJiYgYmF0Y2hTaXplID4gMCkge1xuICAgICAgaWYgKHhbMF0uc2hhcGVbMF0gJSBiYXRjaFNpemUgIT09IDApIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgSW4gYSBzdGF0ZWZ1bCBuZXR3b3JrLCB5b3Ugc2hvdWxkIG9ubHkgcGFzcyBpbnB1dHMgd2l0aCBhIGAgK1xuICAgICAgICAgICAgYG51bWJlciBvZiBzYW1wbGVzIHRoYXQgaXMgZGl2aXNpYmxlIGJ5IHRoZSBiYXRjaCBzaXplIGAgK1xuICAgICAgICAgICAgYCR7YmF0Y2hTaXplfS4gRm91bmQ6ICR7eFswXS5zaGFwZVswXX0gc2FtcGxlKHMpLmApO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gW3gsIHldO1xuICB9XG5cbiAgcHJvdGVjdGVkIGFzeW5jIHN0YW5kYXJkaXplVXNlckRhdGEoXG4gICAgICB4OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgICB5OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgICBzYW1wbGVXZWlnaHQ/OiBUZW5zb3J8VGVuc29yW118e1tvdXRwdXROYW1lOiBzdHJpbmddOiBUZW5zb3J9LFxuICAgICAgY2xhc3NXZWlnaHQ/OiBDbGFzc1dlaWdodHxDbGFzc1dlaWdodFtdfENsYXNzV2VpZ2h0TWFwLFxuICAgICAgY2hlY2tCYXRjaEF4aXMgPSB0cnVlLFxuICAgICAgYmF0Y2hTaXplPzogbnVtYmVyKTogUHJvbWlzZTxbVGVuc29yW10sIFRlbnNvcltdLCBUZW5zb3JbXV0+IHtcbiAgICBjb25zdCBbc3RhbmRhcmRYcywgc3RhbmRhcmRZc10gPVxuICAgICAgICB0aGlzLnN0YW5kYXJkaXplVXNlckRhdGFYWSh4LCB5LCBjaGVja0JhdGNoQXhpcywgYmF0Y2hTaXplKTtcbiAgICAvLyBUT0RPKGNhaXMpOiBIYW5kbGUgc2FtcGxlV2VpZ2h0cy5cbiAgICBpZiAoc2FtcGxlV2VpZ2h0ICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignc2FtcGxlIHdlaWdodCBpcyBub3Qgc3VwcG9ydGVkIHlldC4nKTtcbiAgICB9XG5cbiAgICBsZXQgc3RhbmRhcmRTYW1wbGVXZWlnaHRzOiBUZW5zb3JbXSA9IG51bGw7XG4gICAgaWYgKGNsYXNzV2VpZ2h0ICE9IG51bGwpIHtcbiAgICAgIGNvbnN0IGNsYXNzV2VpZ2h0cyA9XG4gICAgICAgICAgc3RhbmRhcmRpemVDbGFzc1dlaWdodHMoY2xhc3NXZWlnaHQsIHRoaXMub3V0cHV0TmFtZXMpO1xuICAgICAgc3RhbmRhcmRTYW1wbGVXZWlnaHRzID0gW107XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IGNsYXNzV2VpZ2h0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICBzdGFuZGFyZFNhbXBsZVdlaWdodHMucHVzaChcbiAgICAgICAgICAgIGF3YWl0IHN0YW5kYXJkaXplV2VpZ2h0cyhzdGFuZGFyZFlzW2ldLCBudWxsLCBjbGFzc1dlaWdodHNbaV0pKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBUT0RPKGNhaXMpOiBEZWFsIHdpdGggdGhlIGNhc2Ugb2YgbW9kZWwuc3RhdGVmdWwgPT0gdHJ1ZS5cbiAgICByZXR1cm4gW3N0YW5kYXJkWHMsIHN0YW5kYXJkWXMsIHN0YW5kYXJkU2FtcGxlV2VpZ2h0c107XG4gIH1cblxuICAvKipcbiAgICogTG9vcCBvdmVyIHNvbWUgdGVzdCBkYXRhIGluIGJhdGNoZXMuXG4gICAqIEBwYXJhbSBmIEEgRnVuY3Rpb24gcmV0dXJuaW5nIGEgbGlzdCBvZiB0ZW5zb3JzLlxuICAgKiBAcGFyYW0gaW5zIEFycmF5IG9mIHRlbnNvcnMgdG8gYmUgZmVkIHRvIGBmYC5cbiAgICogQHBhcmFtIGJhdGNoU2l6ZSBJbnRlZ2VyIGJhdGNoIHNpemUgb3IgYG51bGxgIC8gYHVuZGVmaW5lZGAuXG4gICAqIEBwYXJhbSB2ZXJib3NlIHZlcmJvc2l0eSBtb2RlLlxuICAgKiBAcGFyYW0gc3RlcHMgVG90YWwgbnVtYmVyIG9mIHN0ZXBzIChiYXRjaGVzIG9mIHNhbXBsZXMpIGJlZm9yZVxuICAgKiBkZWNsYXJpbmcgdGVzdCBmaW5pc2hlZC4gSWdub3JlZCB3aXRoIHRoZSBkZWZhdWx0IHZhbHVlIG9mIGBudWxsYCAvXG4gICAqIGB1bmRlZmluZWRgLlxuICAgKiBAcmV0dXJucyBBcnJheSBvZiBTY2FsYXJzLlxuICAgKi9cbiAgcHJpdmF0ZSB0ZXN0TG9vcChcbiAgICAgIGY6IChkYXRhOiBUZW5zb3JbXSkgPT4gU2NhbGFyW10sIGluczogVGVuc29yW10sIGJhdGNoU2l6ZT86IG51bWJlcixcbiAgICAgIHZlcmJvc2UgPSAwLCBzdGVwcz86IG51bWJlcik6IFNjYWxhcltdIHtcbiAgICByZXR1cm4gdGZjLnRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgbnVtU2FtcGxlcyA9IHRoaXMuY2hlY2tOdW1TYW1wbGVzKGlucywgYmF0Y2hTaXplLCBzdGVwcywgJ3N0ZXBzJyk7XG4gICAgICBjb25zdCBvdXRzOiBTY2FsYXJbXSA9IFtdO1xuICAgICAgaWYgKHZlcmJvc2UgPiAwKSB7XG4gICAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKCdWZXJib3NlIG1vZGUgaXMgbm90IGltcGxlbWVudGVkIHlldC4nKTtcbiAgICAgIH1cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IFVzZSBgaW5kaWNlc0ZvckNvbnZlcnNpb25Ub0RlbnNlJyB0byBwcmV2ZW50IHNsb3cgZG93bi5cbiAgICAgIGlmIChzdGVwcyAhPSBudWxsKSB7XG4gICAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgICAgJ3N0ZXBzIG1vZGUgaW4gdGVzdExvb3AoKSBpcyBub3QgaW1wbGVtZW50ZWQgeWV0Jyk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBjb25zdCBiYXRjaGVzID0gbWFrZUJhdGNoZXMobnVtU2FtcGxlcywgYmF0Y2hTaXplKTtcbiAgICAgICAgY29uc3QgaW5kZXhBcnJheSA9IHRlbnNvcjFkKHJhbmdlKDAsIG51bVNhbXBsZXMpKTtcbiAgICAgICAgZm9yIChsZXQgYmF0Y2hJbmRleCA9IDA7IGJhdGNoSW5kZXggPCBiYXRjaGVzLmxlbmd0aDsgKytiYXRjaEluZGV4KSB7XG4gICAgICAgICAgY29uc3QgYmF0Y2hTdGFydCA9IGJhdGNoZXNbYmF0Y2hJbmRleF1bMF07XG4gICAgICAgICAgY29uc3QgYmF0Y2hFbmQgPSBiYXRjaGVzW2JhdGNoSW5kZXhdWzFdO1xuICAgICAgICAgIGNvbnN0IGJhdGNoSWRzID1cbiAgICAgICAgICAgICAgSy5zbGljZUFsb25nRmlyc3RBeGlzKFxuICAgICAgICAgICAgICAgICAgaW5kZXhBcnJheSwgYmF0Y2hTdGFydCwgYmF0Y2hFbmQgLSBiYXRjaFN0YXJ0KSBhcyBUZW5zb3IxRDtcbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBJbiBpbnMsIHRyYWluIGZsYWcgY2FuIGJlIGEgbnVtYmVyLCBpbnN0ZWFkIG9mIGFuXG4gICAgICAgICAgLy8gICBUZW5zb3I/IERvIHdlIG5lZWQgdG8gaGFuZGxlIHRoaXMgaW4gdGZqcy1sYXllcnM/XG4gICAgICAgICAgY29uc3QgaW5zQmF0Y2ggPSBzbGljZUFycmF5c0J5SW5kaWNlcyhpbnMsIGJhdGNoSWRzKSBhcyBTY2FsYXJbXTtcbiAgICAgICAgICBjb25zdCBiYXRjaE91dHMgPSBmKGluc0JhdGNoKTtcbiAgICAgICAgICBpZiAoYmF0Y2hJbmRleCA9PT0gMCkge1xuICAgICAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBiYXRjaE91dHMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgICAgICAgb3V0cy5wdXNoKHNjYWxhcigwKSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYmF0Y2hPdXRzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgICBjb25zdCBiYXRjaE91dCA9IGJhdGNoT3V0c1tpXTtcbiAgICAgICAgICAgIG91dHNbaV0gPVxuICAgICAgICAgICAgICAgIHRmYy5hZGQob3V0c1tpXSwgdGZjLm11bChiYXRjaEVuZCAtIGJhdGNoU3RhcnQsIGJhdGNoT3V0KSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgb3V0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICAgIG91dHNbaV0gPSB0ZmMuZGl2KG91dHNbaV0sIG51bVNhbXBsZXMpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICByZXR1cm4gb3V0cztcbiAgICB9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBnZXREZWR1cGVkTWV0cmljc05hbWVzKCk6IHN0cmluZ1tdIHtcbiAgICBjb25zdCBvdXRMYWJlbHMgPSB0aGlzLm1ldHJpY3NOYW1lcztcbiAgICAvLyBSZW5hbWUgZHVwbGljYXRlZCBtZXRyaWNzIG5hbWVzIChjYW4gaGFwcGVuIHdpdGggYW4gb3V0cHV0IGxheWVyXG4gICAgLy8gc2hhcmVkIGFtb25nIG11bHRpcGxlIGRhdGFmbG93cykuXG4gICAgY29uc3QgZGVkdXBlZE91dExhYmVscyA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgb3V0TGFiZWxzLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCBsYWJlbCA9IG91dExhYmVsc1tpXTtcbiAgICAgIGxldCBuZXdMYWJlbCA9IGxhYmVsO1xuICAgICAgaWYgKGNvdW50KG91dExhYmVscywgbGFiZWwpID4gMSkge1xuICAgICAgICBjb25zdCBkdXBJbmRleCA9IGNvdW50KG91dExhYmVscy5zbGljZSgwLCBpKSwgbGFiZWwpO1xuICAgICAgICBuZXdMYWJlbCArPSBgXyR7ZHVwSW5kZXh9YDtcbiAgICAgIH1cbiAgICAgIGRlZHVwZWRPdXRMYWJlbHMucHVzaChuZXdMYWJlbCk7XG4gICAgfVxuICAgIHJldHVybiBkZWR1cGVkT3V0TGFiZWxzO1xuICB9XG5cbiAgLyoqXG4gICAqIENyZWF0ZXMgYSBmdW5jdGlvbiB0aGF0IHBlcmZvcm1zIHRoZSBmb2xsb3dpbmcgYWN0aW9uczpcbiAgICpcbiAgICogMS4gY29tcHV0ZXMgdGhlIGxvc3Nlc1xuICAgKiAyLiBzdW1zIHRoZW0gdG8gZ2V0IHRoZSB0b3RhbCBsb3NzXG4gICAqIDMuIGNhbGwgdGhlIG9wdGltaXplciBjb21wdXRlcyB0aGUgZ3JhZGllbnRzIG9mIHRoZSBMYXllcnNNb2RlbCdzXG4gICAqICAgIHRyYWluYWJsZSB3ZWlnaHRzIHcuci50LiB0aGUgdG90YWwgbG9zcyBhbmQgdXBkYXRlIHRoZSB2YXJpYWJsZXNcbiAgICogNC4gY2FsY3VsYXRlcyB0aGUgbWV0cmljc1xuICAgKiA1LiByZXR1cm5zIHRoZSB2YWx1ZXMgb2YgdGhlIGxvc3NlcyBhbmQgbWV0cmljcy5cbiAgICovXG4gIHByb3RlY3RlZCBtYWtlVHJhaW5GdW5jdGlvbigpOiAoZGF0YTogVGVuc29yW10pID0+IFNjYWxhcltdIHtcbiAgICByZXR1cm4gKGRhdGE6IFRlbnNvcltdKSA9PiB7XG4gICAgICBjb25zdCBsb3NzVmFsdWVzOiBTY2FsYXJbXSA9IFtdO1xuXG4gICAgICBjb25zdCBpbnB1dHMgPSBkYXRhLnNsaWNlKDAsIHRoaXMuaW5wdXRzLmxlbmd0aCk7XG4gICAgICBjb25zdCB0YXJnZXRzID0gZGF0YS5zbGljZShcbiAgICAgICAgICB0aGlzLmlucHV0cy5sZW5ndGgsIHRoaXMuaW5wdXRzLmxlbmd0aCArIHRoaXMub3V0cHV0cy5sZW5ndGgpO1xuICAgICAgY29uc3Qgc2FtcGxlV2VpZ2h0cyA9IGRhdGEuc2xpY2UoXG4gICAgICAgICAgdGhpcy5pbnB1dHMubGVuZ3RoICsgdGhpcy5vdXRwdXRzLmxlbmd0aCxcbiAgICAgICAgICB0aGlzLmlucHV0cy5sZW5ndGggKyB0aGlzLm91dHB1dHMubGVuZ3RoICogMik7XG5cbiAgICAgIGNvbnN0IG1ldHJpY3NWYWx1ZXM6IFNjYWxhcltdID0gW107XG5cbiAgICAgIC8vIENyZWF0ZSBhIGZ1bmN0aW9uIHRoYXQgY29tcHV0ZXMgdGhlIHRvdGFsIGxvc3MgYmFzZWQgb24gdGhlXG4gICAgICAvLyBpbnB1dHMuIFRoaXMgZnVuY3Rpb24gaXMgdXNlZCBmb3Igb2J0YWluaW5nIGdyYWRpZW50cyB0aHJvdWdoXG4gICAgICAvLyBiYWNrcHJvcC5cbiAgICAgIGNvbnN0IHRvdGFsTG9zc0Z1bmN0aW9uID0gKCkgPT4ge1xuICAgICAgICBjb25zdCBmZWVkcyA9IFtdO1xuICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuaW5wdXRzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgZmVlZHMucHVzaCh7a2V5OiB0aGlzLmlucHV0c1tpXSwgdmFsdWU6IGlucHV0c1tpXX0pO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGZlZWREaWN0ID0gbmV3IEZlZWREaWN0KGZlZWRzKTtcbiAgICAgICAgY29uc3Qgb3V0cHV0cyA9XG4gICAgICAgICAgICBleGVjdXRlKHRoaXMub3V0cHV0cywgZmVlZERpY3QsIHsndHJhaW5pbmcnOiB0cnVlfSkgYXMgVGVuc29yW107XG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IFRha2UgY2FyZSBvZiB0aGUgY2FzZSBvZiBtdWx0aXBsZSBvdXRwdXRzIGZyb20gYVxuICAgICAgICAvLyAgIHNpbmdsZSBsYXllcj9cblxuICAgICAgICBsZXQgdG90YWxMb3NzOiBUZW5zb3I7XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5sb3NzRnVuY3Rpb25zLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgY29uc3QgbG9zc0Z1bmN0aW9uID0gdGhpcy5sb3NzRnVuY3Rpb25zW2ldO1xuICAgICAgICAgIGxldCBsb3NzID0gbG9zc0Z1bmN0aW9uKHRhcmdldHNbaV0sIG91dHB1dHNbaV0pO1xuICAgICAgICAgIGlmIChzYW1wbGVXZWlnaHRzW2ldICE9IG51bGwpIHtcbiAgICAgICAgICAgIGxvc3MgPSBjb21wdXRlV2VpZ2h0ZWRMb3NzKGxvc3MsIHNhbXBsZVdlaWdodHNbaV0pO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IHB1c2ggU2NhbGFyIGluc3RlYWQuXG4gICAgICAgICAgY29uc3QgbWVhbkxvc3M6IFNjYWxhciA9IHRmYy5tZWFuKGxvc3MpO1xuICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IFVzZSBhIHNjb3BlKCkgaW5zdGVhZCwgdG8gYXZvaWQgb3duZXJzaGlwLlxuICAgICAgICAgIGxvc3NWYWx1ZXMucHVzaChtZWFuTG9zcyk7XG4gICAgICAgICAgaWYgKGkgPT09IDApIHtcbiAgICAgICAgICAgIHRvdGFsTG9zcyA9IGxvc3M7XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHRvdGFsTG9zcyA9IHRmYy5hZGQodG90YWxMb3NzLCBsb3NzKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cblxuICAgICAgICAvLyBDb21wdXRlIHRoZSBtZXRyaWNzLlxuICAgICAgICAvLyBUT0RPKGNhaXMpOiBUaGVzZSBzaG91bGQgcHJvYmFibHkgYmUgY2FsY3VsYXRlZCBvdXRzaWRlXG4gICAgICAgIC8vICAgdG90YWxMb3NzRnVuY3Rpb24gdG8gYmVuZWZpdCBzcGVlZD9cbiAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLm1ldHJpY3NUZW5zb3JzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgbGV0IHdlaWdodGVkTWV0cmljOiBTY2FsYXI7XG5cbiAgICAgICAgICBpZiAodGhpcy5vdXRwdXRzLmxlbmd0aCA+IDEgJiYgaSA8IHRoaXMub3V0cHV0cy5sZW5ndGgpIHtcbiAgICAgICAgICAgIHdlaWdodGVkTWV0cmljID0gbG9zc1ZhbHVlc1tpXTtcbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgY29uc3QgbWV0cmljID0gdGhpcy5tZXRyaWNzVGVuc29yc1tpXVswXTtcbiAgICAgICAgICAgIGNvbnN0IG91dHB1dEluZGV4ID0gdGhpcy5tZXRyaWNzVGVuc29yc1tpXVsxXTtcbiAgICAgICAgICAgIHdlaWdodGVkTWV0cmljID1cbiAgICAgICAgICAgICAgICB0ZmMubWVhbihtZXRyaWModGFyZ2V0c1tvdXRwdXRJbmRleF0sIG91dHB1dHNbb3V0cHV0SW5kZXhdKSk7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgdGZjLmtlZXAod2VpZ2h0ZWRNZXRyaWMpO1xuICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IFVzZSBhIHNjb3BlKCkgaW5zdGVhZCwgdG8gYXZvaWQgb3duZXJzaGlwLlxuICAgICAgICAgIG1ldHJpY3NWYWx1ZXMucHVzaCh3ZWlnaHRlZE1ldHJpYyk7XG4gICAgICAgIH1cblxuICAgICAgICB0b3RhbExvc3MgPSB0ZmMubWVhbih0b3RhbExvc3MpO1xuXG4gICAgICAgIC8vIEFkZCByZWd1bGFyaXplciBwZW5hbHRpZXMuXG4gICAgICAgIHRoaXMuY2FsY3VsYXRlTG9zc2VzKCkuZm9yRWFjaChyZWd1bGFyaXplckxvc3MgPT4ge1xuICAgICAgICAgIHRvdGFsTG9zcyA9IHRmYy5hZGQodG90YWxMb3NzLCByZWd1bGFyaXplckxvc3MpO1xuICAgICAgICB9KTtcblxuICAgICAgICByZXR1cm4gdG90YWxMb3NzIGFzIFNjYWxhcjtcbiAgICAgIH07XG5cbiAgICAgIGNvbnN0IHZhcmlhYmxlcyA9IHRoaXMuY29sbGVjdGVkVHJhaW5hYmxlV2VpZ2h0cy5tYXAoXG4gICAgICAgICAgcGFyYW0gPT4gcGFyYW0ucmVhZCgpIGFzIHRmYy5WYXJpYWJsZSk7XG4gICAgICBjb25zdCByZXR1cm5Db3N0ID0gdHJ1ZTtcbiAgICAgIGNvbnN0IHRvdGFsTG9zc1ZhbHVlID1cbiAgICAgICAgICB0aGlzLm9wdGltaXplcl8ubWluaW1pemUodG90YWxMb3NzRnVuY3Rpb24sIHJldHVybkNvc3QsIHZhcmlhYmxlcyk7XG5cbiAgICAgIHJldHVybiBbdG90YWxMb3NzVmFsdWVdLmNvbmNhdChtZXRyaWNzVmFsdWVzKTtcbiAgICB9O1xuICB9XG5cbiAgLyoqXG4gICAqIENyZWF0ZSBhIGZ1bmN0aW9uIHdoaWNoLCB3aGVuIGludm9rZWQgd2l0aCBhbiBhcnJheSBvZiBgdGYuVGVuc29yYHMgYXMgYVxuICAgKiBiYXRjaCBvZiBpbnB1dHMsIHJldHVybnMgdGhlIHByZXNwZWNpZmllZCBsb3NzIGFuZCBtZXRyaWNzIG9mIHRoZSBtb2RlbFxuICAgKiB1bmRlciB0aGUgYmF0Y2ggb2YgaW5wdXQgZGF0YS5cbiAgICovXG4gIHByaXZhdGUgbWFrZVRlc3RGdW5jdGlvbigpIHtcbiAgICB0aGlzLnRlc3RGdW5jdGlvbiA9IChkYXRhOiBUZW5zb3JbXSkgPT4ge1xuICAgICAgcmV0dXJuIHRmYy50aWR5KCgpID0+IHtcbiAgICAgICAgY29uc3QgdmFsT3V0cHV0czogU2NhbGFyW10gPSBbXTtcbiAgICAgICAgbGV0IHRvdGFsTG9zczogU2NhbGFyO1xuICAgICAgICBjb25zdCBpbnB1dHMgPSBkYXRhLnNsaWNlKDAsIHRoaXMuaW5wdXRzLmxlbmd0aCk7XG4gICAgICAgIGNvbnN0IHRhcmdldHMgPSBkYXRhLnNsaWNlKFxuICAgICAgICAgICAgdGhpcy5pbnB1dHMubGVuZ3RoLCB0aGlzLmlucHV0cy5sZW5ndGggKyB0aGlzLm91dHB1dHMubGVuZ3RoKTtcbiAgICAgICAgY29uc3QgZmVlZHMgPSBbXTtcbiAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLmlucHV0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICAgIGZlZWRzLnB1c2goe2tleTogdGhpcy5pbnB1dHNbaV0sIHZhbHVlOiBpbnB1dHNbaV19KTtcbiAgICAgICAgfVxuICAgICAgICBjb25zdCBmZWVkRGljdCA9IG5ldyBGZWVkRGljdChmZWVkcyk7XG4gICAgICAgIGNvbnN0IG91dHB1dHMgPSBleGVjdXRlKHRoaXMub3V0cHV0cywgZmVlZERpY3QpIGFzIFRlbnNvcltdO1xuICAgICAgICAvLyBDb21wdXRlIHRvdGFsIGxvc3MuXG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5sb3NzRnVuY3Rpb25zLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgY29uc3QgbG9zc0Z1bmN0aW9uID0gdGhpcy5sb3NzRnVuY3Rpb25zW2ldO1xuICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBzYW1wbGUgd2VpZ2h0aW5nIGFuZCByZXBsYWNlIHRoZSBzaW1wbGVcbiAgICAgICAgICAvLyBhdmVyYWdpbmcuXG4gICAgICAgICAgY29uc3QgbG9zczogU2NhbGFyID0gdGZjLm1lYW4obG9zc0Z1bmN0aW9uKHRhcmdldHNbaV0sIG91dHB1dHNbaV0pKTtcbiAgICAgICAgICBpZiAoaSA9PT0gMCkge1xuICAgICAgICAgICAgdG90YWxMb3NzID0gbG9zcztcbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgdG90YWxMb3NzID0gdGZjLmFkZCh0b3RhbExvc3MsIGxvc3MpO1xuICAgICAgICAgIH1cbiAgICAgICAgICB2YWxPdXRwdXRzLnB1c2godG90YWxMb3NzKTtcbiAgICAgICAgfVxuICAgICAgICAvLyBDb21wdXRlIHRoZSBtZXRyaWNzLlxuICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMubWV0cmljc1RlbnNvcnMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgICBjb25zdCBtZXRyaWMgPSB0aGlzLm1ldHJpY3NUZW5zb3JzW2ldWzBdO1xuICAgICAgICAgIGNvbnN0IG91dHB1dEluZGV4ID0gdGhpcy5tZXRyaWNzVGVuc29yc1tpXVsxXTtcbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBSZXBsYWNlIEsubWVhbigpIHdpdGggYSBwcm9wZXIgd2VpZ2h0aW5nIGZ1bmN0aW9uLlxuICAgICAgICAgIGNvbnN0IG1lYW5NZXRyaWMgPVxuICAgICAgICAgICAgICB0ZmMubWVhbihtZXRyaWModGFyZ2V0c1tvdXRwdXRJbmRleF0sIG91dHB1dHNbb3V0cHV0SW5kZXhdKSk7XG4gICAgICAgICAgdmFsT3V0cHV0cy5wdXNoKG1lYW5NZXRyaWMgYXMgU2NhbGFyKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gdmFsT3V0cHV0cztcbiAgICAgIH0pO1xuICAgIH07XG4gIH1cblxuICAvKipcbiAgICogVHJhaW5zIHRoZSBtb2RlbCBmb3IgYSBmaXhlZCBudW1iZXIgb2YgZXBvY2hzIChpdGVyYXRpb25zIG9uIGFcbiAgICogZGF0YXNldCkuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCh7XG4gICAqICAgICBsYXllcnM6IFt0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbMTBdfSldXG4gICAqIH0pO1xuICAgKiBtb2RlbC5jb21waWxlKHtvcHRpbWl6ZXI6ICdzZ2QnLCBsb3NzOiAnbWVhblNxdWFyZWRFcnJvcid9KTtcbiAgICogZm9yIChsZXQgaSA9IDE7IGkgPCA1IDsgKytpKSB7XG4gICAqICAgY29uc3QgaCA9IGF3YWl0IG1vZGVsLmZpdCh0Zi5vbmVzKFs4LCAxMF0pLCB0Zi5vbmVzKFs4LCAxXSksIHtcbiAgICogICAgICAgYmF0Y2hTaXplOiA0LFxuICAgKiAgICAgICBlcG9jaHM6IDNcbiAgICogICB9KTtcbiAgICogICBjb25zb2xlLmxvZyhcIkxvc3MgYWZ0ZXIgRXBvY2ggXCIgKyBpICsgXCIgOiBcIiArIGguaGlzdG9yeS5sb3NzWzBdKTtcbiAgICogfVxuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIHggYHRmLlRlbnNvcmAgb2YgdHJhaW5pbmcgZGF0YSwgb3IgYW4gYXJyYXkgb2YgYHRmLlRlbnNvcmBzIGlmIHRoZVxuICAgKiBtb2RlbCBoYXMgbXVsdGlwbGUgaW5wdXRzLiBJZiBhbGwgaW5wdXRzIGluIHRoZSBtb2RlbCBhcmUgbmFtZWQsIHlvdVxuICAgKiBjYW4gYWxzbyBwYXNzIGEgZGljdGlvbmFyeSBtYXBwaW5nIGlucHV0IG5hbWVzIHRvIGB0Zi5UZW5zb3Jgcy5cbiAgICogQHBhcmFtIHkgYHRmLlRlbnNvcmAgb2YgdGFyZ2V0IChsYWJlbCkgZGF0YSwgb3IgYW4gYXJyYXkgb2YgYHRmLlRlbnNvcmBzIGlmXG4gICAqIHRoZSBtb2RlbCBoYXMgbXVsdGlwbGUgb3V0cHV0cy4gSWYgYWxsIG91dHB1dHMgaW4gdGhlIG1vZGVsIGFyZSBuYW1lZCxcbiAgICogeW91IGNhbiBhbHNvIHBhc3MgYSBkaWN0aW9uYXJ5IG1hcHBpbmcgb3V0cHV0IG5hbWVzIHRvIGB0Zi5UZW5zb3Jgcy5cbiAgICogQHBhcmFtIGFyZ3MgQSBgTW9kZWxGaXRBcmdzYCwgY29udGFpbmluZyBvcHRpb25hbCBmaWVsZHMuXG4gICAqXG4gICAqIEByZXR1cm4gQSBgSGlzdG9yeWAgaW5zdGFuY2UuIEl0cyBgaGlzdG9yeWAgYXR0cmlidXRlIGNvbnRhaW5zIGFsbFxuICAgKiAgIGluZm9ybWF0aW9uIGNvbGxlY3RlZCBkdXJpbmcgdHJhaW5pbmcuXG4gICAqXG4gICAqIEBleGNlcHRpb24gVmFsdWVFcnJvciBJbiBjYXNlIG9mIG1pc21hdGNoIGJldHdlZW4gdGhlIHByb3ZpZGVkIGlucHV0XG4gICAqIGRhdGEgYW5kIHdoYXQgdGhlIG1vZGVsIGV4cGVjdHMuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBhc3luYyBmaXQoXG4gICAgICB4OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgICB5OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgICBhcmdzOiBNb2RlbEZpdEFyZ3MgPSB7fSk6IFByb21pc2U8SGlzdG9yeT4ge1xuICAgIGlmICh0aGlzLmlzVHJhaW5pbmcpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnQ2Fubm90IHN0YXJ0IHRyYWluaW5nIGJlY2F1c2UgYW5vdGhlciBmaXQoKSBjYWxsIGlzIG9uZ29pbmcuJyk7XG4gICAgfVxuICAgIHRoaXMuaXNUcmFpbmluZyA9IHRydWU7XG4gICAgbGV0IGlucHV0czogVGVuc29yW107XG4gICAgbGV0IHRhcmdldHM6IFRlbnNvcltdO1xuICAgIGxldCBvcmlnaW5hbElucHV0czogVGVuc29yW107XG4gICAgbGV0IG9yaWdpbmFsVGFyZ2V0czogVGVuc29yW107XG4gICAgbGV0IGlucHV0VmFsWDogVGVuc29yfFRlbnNvcltdO1xuICAgIGxldCBpbnB1dFZhbFk6IFRlbnNvcnxUZW5zb3JbXTtcbiAgICBsZXQgdmFsWDogVGVuc29yfFRlbnNvcltdO1xuICAgIGxldCB2YWxZOiBUZW5zb3J8VGVuc29yW107XG4gICAgbGV0IHNhbXBsZVdlaWdodHM6IFRlbnNvcltdO1xuICAgIHRyeSB7XG4gICAgICBjb25zdCBiYXRjaFNpemUgPSBhcmdzLmJhdGNoU2l6ZSA9PSBudWxsID8gMzIgOiBhcmdzLmJhdGNoU2l6ZTtcbiAgICAgIGNoZWNrQmF0Y2hTaXplKGJhdGNoU2l6ZSk7XG5cbiAgICAgIC8vIFZhbGlkYXRlIHVzZXIgZGF0YS5cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IFN1cHBvcnQgc2FtcGxlV2VpZ2h0LlxuICAgICAgY29uc3QgY2hlY2tCYXRjaEF4aXMgPSBmYWxzZTtcbiAgICAgIGNvbnN0IHN0YW5kYXJkaXplZE91dHMgPVxuICAgICAgICAgIGF3YWl0IHRoaXMuc3RhbmRhcmRpemVVc2VyRGF0YShcbiAgICAgICAgICAgICAgeCwgeSwgYXJncy5zYW1wbGVXZWlnaHQsIGFyZ3MuY2xhc3NXZWlnaHQsIGNoZWNrQmF0Y2hBeGlzLFxuICAgICAgICAgICAgICBiYXRjaFNpemUpIGFzIFtUZW5zb3JbXSwgVGVuc29yW10sIFRlbnNvcltdXTtcbiAgICAgIGlucHV0cyA9IHN0YW5kYXJkaXplZE91dHNbMF07XG4gICAgICB0YXJnZXRzID0gc3RhbmRhcmRpemVkT3V0c1sxXTtcbiAgICAgIHNhbXBsZVdlaWdodHMgPSBzdGFuZGFyZGl6ZWRPdXRzWzJdO1xuXG4gICAgICAvLyBQcmVwYXJlIHZhbGlkYXRpb24gZGF0YS5cbiAgICAgIGxldCBkb1ZhbGlkYXRpb24gPSBmYWxzZTtcbiAgICAgIGxldCB2YWxJbnM6IFRlbnNvcltdO1xuICAgICAgaWYgKGFyZ3MudmFsaWRhdGlvbkRhdGEgIT0gbnVsbCAmJiBhcmdzLnZhbGlkYXRpb25EYXRhLmxlbmd0aCA+IDApIHtcbiAgICAgICAgZG9WYWxpZGF0aW9uID0gdHJ1ZTtcbiAgICAgICAgaWYgKGFyZ3MudmFsaWRhdGlvbkRhdGEubGVuZ3RoID09PSAyKSB7XG4gICAgICAgICAgLy8gY29uZmlnLnZhbGlkYXRpb25EYXRhIGNvbnNpc3RzIG9mIHZhbFggYW5kIHZhbFkuXG4gICAgICAgICAgaW5wdXRWYWxYID0gYXJncy52YWxpZGF0aW9uRGF0YVswXTtcbiAgICAgICAgICBpbnB1dFZhbFkgPSBhcmdzLnZhbGlkYXRpb25EYXRhWzFdO1xuICAgICAgICB9IGVsc2UgaWYgKGFyZ3MudmFsaWRhdGlvbkRhdGEubGVuZ3RoID09PSAzKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgICAgICd2YWxpZGF0aW9uRGF0YSBpbmNsdWRpbmcgc2FtcGxlIHdlaWdodHMgaXMgbm90IHN1cHBvcnRlZCB5ZXQuJyk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGBXaGVuIHBhc3NpbmcgdmFsaWRhdGlvbiBkYXRhLCBpdCBtdXN0IGNvbnRhaW4gMiAodmFsWCwgdmFsWSkgYCArXG4gICAgICAgICAgICAgIGBvciAzICh2YWxYLCB2YWxZLCB2YWxTYW1wbGVXZWlnaHQpIGl0ZW1zOyBgICtcbiAgICAgICAgICAgICAgYCR7YXJncy52YWxpZGF0aW9uRGF0YX0gaXMgaW52YWxpZC5gKTtcbiAgICAgICAgfVxuXG4gICAgICAgIGNvbnN0IGNoZWNrQmF0Y2hBeGlzID0gdHJ1ZTtcbiAgICAgICAgY29uc3QgdmFsU3RhbmRhcmRpemVkID1cbiAgICAgICAgICAgIGF3YWl0IHRoaXMuc3RhbmRhcmRpemVVc2VyRGF0YShcbiAgICAgICAgICAgICAgICBpbnB1dFZhbFgsIGlucHV0VmFsWSwgbnVsbCwgLyoqIFVudXNlZCBzYW1wbGUgd2VpZ2h0cy4gKi9cbiAgICAgICAgICAgICAgICBudWxsLCAgICAgICAgICAgICAgICAgICAgICAgLyoqIFVudXNlZCBjbGFzcyB3ZWlnaHRzLiAqL1xuICAgICAgICAgICAgICAgIGNoZWNrQmF0Y2hBeGlzLCBiYXRjaFNpemUpIGFzIFtUZW5zb3JbXSwgVGVuc29yW10sIFRlbnNvcltdXTtcbiAgICAgICAgdmFsWCA9IHZhbFN0YW5kYXJkaXplZFswXTtcbiAgICAgICAgdmFsWSA9IHZhbFN0YW5kYXJkaXplZFsxXTtcbiAgICAgICAgdmFsSW5zID0gdmFsWC5jb25jYXQodmFsWSk7XG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCB1c2VMZWFybmluZ1BoYXNlIGRhdGEgcHJvcGVybHkuXG4gICAgICB9IGVsc2UgaWYgKFxuICAgICAgICAgIGFyZ3MudmFsaWRhdGlvblNwbGl0ICE9IG51bGwgJiYgYXJncy52YWxpZGF0aW9uU3BsaXQgPiAwICYmXG4gICAgICAgICAgYXJncy52YWxpZGF0aW9uU3BsaXQgPCAxKSB7XG4gICAgICAgIGRvVmFsaWRhdGlvbiA9IHRydWU7XG4gICAgICAgIC8vIFBvcnRpbmcgTm90ZTogSW4gdGZqcy1sYXllcnMsIGlucHV0c1swXSBpcyBhbHdheXMgYSBUZW5zb3IuXG4gICAgICAgIGNvbnN0IHNwbGl0QXQgPVxuICAgICAgICAgICAgTWF0aC5mbG9vcihpbnB1dHNbMF0uc2hhcGVbMF0gKiAoMSAtIGFyZ3MudmFsaWRhdGlvblNwbGl0KSk7XG4gICAgICAgIGNvbnN0IG9yaWdpbmFsQmF0Y2hTaXplID0gaW5wdXRzWzBdLnNoYXBlWzBdO1xuICAgICAgICB2YWxYID0gc2xpY2VBcnJheXMoaW5wdXRzLCBzcGxpdEF0LCBvcmlnaW5hbEJhdGNoU2l6ZSkgYXMgVGVuc29yW107XG4gICAgICAgIG9yaWdpbmFsSW5wdXRzID0gaW5wdXRzO1xuICAgICAgICBpbnB1dHMgPSBzbGljZUFycmF5cyhpbnB1dHMsIDAsIHNwbGl0QXQpIGFzIFRlbnNvcltdO1xuICAgICAgICB2YWxZID0gc2xpY2VBcnJheXModGFyZ2V0cywgc3BsaXRBdCwgb3JpZ2luYWxCYXRjaFNpemUpIGFzIFRlbnNvcltdO1xuICAgICAgICBvcmlnaW5hbFRhcmdldHMgPSB0YXJnZXRzO1xuICAgICAgICB0YXJnZXRzID0gc2xpY2VBcnJheXModGFyZ2V0cywgMCwgc3BsaXRBdCkgYXMgVGVuc29yW107XG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IE9uY2Ugc2FtcGxlV2VpZ2h0cyBiZWNvbWVzIGF2YWlsYWJsZSwgc2xpY2UgaXQgdG8gZ2V0XG4gICAgICAgIC8vICAgdmFsU2FtcGxlV2VpZ2h0cy5cbiAgICAgICAgdmFsSW5zID0gdmFsWC5jb25jYXQodmFsWSk7XG5cbiAgICAgICAgLy8gVE9ETyhjYWlzKTogQWRkIHVzZUxlYXJuaW5nUGhhc2UgZGF0YSBwcm9wZXJseS5cbiAgICAgIH0gZWxzZSBpZiAoYXJncy52YWxpZGF0aW9uU3RlcHMgIT0gbnVsbCkge1xuICAgICAgICBkb1ZhbGlkYXRpb24gPSB0cnVlO1xuICAgICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgdXNlTGVhcm5pbmdQaGFzZS5cbiAgICAgIH1cblxuICAgICAgY29uc3QgaW5zID0gaW5wdXRzLmNvbmNhdCh0YXJnZXRzKS5jb25jYXQoc2FtcGxlV2VpZ2h0cyk7XG5cbiAgICAgIHRoaXMuY2hlY2tUcmFpbmFibGVXZWlnaHRzQ29uc2lzdGVuY3koKTtcblxuICAgICAgLy8gVE9ETyhjYWlzKTogSGFuZGxlIHVzZV9sZWFybmluZ19waGFzZSBhbmQgbGVhcm5pbmdfcGhhc2U/XG5cbiAgICAgIC8vIFBvcnRpbmcgTm90ZTogSGVyZSB3ZSBzZWUgYSBrZXkgZGV2aWF0aW9uIG9mIHRmanMtbGF5ZXJzIGZyb21cbiAgICAgIC8vIEtlcmFzLlxuICAgICAgLy8gIER1ZSB0byB0aGUgaW1wZXJhdGl2ZSBuYXR1cmUgb2YgdGZqcy1sYXllcnMnIGJhY2tlbmQgKHRmanMtY29yZSksXG4gICAgICAvLyAgd2UgZG8gbm90IGNvbnN0cnVjdCBzeW1ib2xpYyBjb21wdXRhdGlvbiBncmFwaHMgdG8gZW1ib2R5IHRoZVxuICAgICAgLy8gIHRyYWluaW5nIHByb2Nlc3MuIEluc3RlYWQsIHdlIGRlZmluZSBhIGZ1bmN0aW9uIHRoYXQgcGVyZm9ybXMgdGhlXG4gICAgICAvLyAgdHJhaW5pbmcgYWN0aW9uLiBJbiBQeUtlcmFzLCB0aGUgZGF0YSAoaW5wdXRzIGFuZCB0YXJnZXRzKSBhcmUgZmVkXG4gICAgICAvLyAgdGhyb3VnaCBncmFwaCBwbGFjZWhvbGRlcnMuIEluIHRmanMtbGF5ZXJzLCB0aGUgZGF0YSBhcmUgZmVkIGFzXG4gICAgICAvLyAgZnVuY3Rpb24gYXJndW1lbnRzLiBTaW5jZSB0aGUgZnVuY3Rpb24gYXJlIGRlZmluZWQgYmVsb3cgaW4gdGhlXG4gICAgICAvLyAgc2NvcGUsIHdlIGRvbid0IGhhdmUgZXF1aXZhbGVudHMgb2YgUHlLZXJhcydzXG4gICAgICAvLyAgYF9tYWtlX3RyYWluX2Z1bmNpdG9uYC5cbiAgICAgIGNvbnN0IHRyYWluRnVuY3Rpb24gPSB0aGlzLm1ha2VUcmFpbkZ1bmN0aW9uKCk7XG4gICAgICBjb25zdCBvdXRMYWJlbHMgPSB0aGlzLmdldERlZHVwZWRNZXRyaWNzTmFtZXMoKTtcblxuICAgICAgbGV0IHZhbEZ1bmN0aW9uOiAoZGF0YTogVGVuc29yW10pID0+IFNjYWxhcltdO1xuICAgICAgbGV0IGNhbGxiYWNrTWV0cmljczogc3RyaW5nW107XG4gICAgICBpZiAoZG9WYWxpZGF0aW9uKSB7XG4gICAgICAgIHRoaXMubWFrZVRlc3RGdW5jdGlvbigpO1xuICAgICAgICB2YWxGdW5jdGlvbiA9IHRoaXMudGVzdEZ1bmN0aW9uO1xuICAgICAgICBjYWxsYmFja01ldHJpY3MgPVxuICAgICAgICAgICAgb3V0TGFiZWxzLnNsaWNlKCkuY29uY2F0KG91dExhYmVscy5tYXAobiA9PiAndmFsXycgKyBuKSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB2YWxGdW5jdGlvbiA9IG51bGw7XG4gICAgICAgIHZhbElucyA9IFtdO1xuICAgICAgICBjYWxsYmFja01ldHJpY3MgPSBvdXRMYWJlbHMuc2xpY2UoKTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgY2FsbGJhY2tzID0gc3RhbmRhcmRpemVDYWxsYmFja3MoYXJncy5jYWxsYmFja3MsIGFyZ3MueWllbGRFdmVyeSk7XG4gICAgICBjb25zdCBvdXQgPSBhd2FpdCB0aGlzLmZpdExvb3AoXG4gICAgICAgICAgdHJhaW5GdW5jdGlvbiwgaW5zLCBvdXRMYWJlbHMsIGJhdGNoU2l6ZSwgYXJncy5lcG9jaHMsXG4gICAgICAgICAgYXJncy52ZXJib3NlLCBjYWxsYmFja3MsIHZhbEZ1bmN0aW9uLCB2YWxJbnMsIGFyZ3Muc2h1ZmZsZSxcbiAgICAgICAgICBjYWxsYmFja01ldHJpY3MsIGFyZ3MuaW5pdGlhbEVwb2NoLCBudWxsLCBudWxsKTtcbiAgICAgIHJldHVybiBvdXQ7XG4gICAgfSBmaW5hbGx5IHtcbiAgICAgIHRoaXMuaXNUcmFpbmluZyA9IGZhbHNlO1xuICAgICAgLy8gTWVtb3J5IGNsZWFuIHVwLlxuICAgICAgZGlzcG9zZU5ld1RlbnNvcnMoaW5wdXRzLCB4KTtcbiAgICAgIGRpc3Bvc2VOZXdUZW5zb3JzKHRhcmdldHMsIHkpO1xuICAgICAgZGlzcG9zZU5ld1RlbnNvcnMob3JpZ2luYWxJbnB1dHMsIHgpO1xuICAgICAgZGlzcG9zZU5ld1RlbnNvcnMob3JpZ2luYWxUYXJnZXRzLCB5KTtcbiAgICAgIGRpc3Bvc2VOZXdUZW5zb3JzKHZhbFggYXMgVGVuc29yW10sIGlucHV0VmFsWCk7XG4gICAgICBkaXNwb3NlTmV3VGVuc29ycyh2YWxZIGFzIFRlbnNvcltdLCBpbnB1dFZhbFkpO1xuICAgICAgaWYgKHNhbXBsZVdlaWdodHMgIT0gbnVsbCkge1xuICAgICAgICB0ZmMuZGlzcG9zZShzYW1wbGVXZWlnaHRzKTtcbiAgICAgIH1cbiAgICB9XG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIHZhbHVlIHRvIG91dExhYmVscy5cbiAgfVxuXG4gIC8qKlxuICAgKiBBYnN0cmFjdCBmaXQgZnVuY3Rpb24gZm9yIGBmKGlucylgLlxuICAgKiBAcGFyYW0gZiBBIEZ1bmN0aW9uIHJldHVybmluZyBhIGxpc3Qgb2YgdGVuc29ycy4gRm9yIHRyYWluaW5nLCB0aGlzXG4gICAqICAgZnVuY3Rpb24gaXMgZXhwZWN0ZWQgdG8gcGVyZm9ybSB0aGUgdXBkYXRlcyB0byB0aGUgdmFyaWFibGVzLlxuICAgKiBAcGFyYW0gaW5zIExpc3Qgb2YgdGVuc29ycyB0byBiZSBmZWQgdG8gYGZgLlxuICAgKiBAcGFyYW0gb3V0TGFiZWxzIExpc3Qgb2Ygc3RyaW5ncywgZGlzcGxheSBuYW1lcyBvZiB0aGUgb3V0cHV0cyBvZiBgZmAuXG4gICAqIEBwYXJhbSBiYXRjaFNpemUgSW50ZWdlciBiYXRjaCBzaXplIG9yIGA9PSBudWxsYCBpZiB1bmtub3duLiBEZWZhdWx0IDogMzIuXG4gICAqIEBwYXJhbSBlcG9jaHMgTnVtYmVyIG9mIHRpbWVzIHRvIGl0ZXJhdGUgb3ZlciB0aGUgZGF0YS4gRGVmYXVsdCA6IDEuXG4gICAqIEBwYXJhbSB2ZXJib3NlIFZlcmJvc2l0eSBtb2RlOiAwLCAxLCBvciAyLiBEZWZhdWx0OiAxLlxuICAgKiBAcGFyYW0gY2FsbGJhY2tzIExpc3Qgb2YgY2FsbGJhY2tzIHRvIGJlIGNhbGxlZCBkdXJpbmcgdHJhaW5pbmcuXG4gICAqIEBwYXJhbSB2YWxGIEZ1bmN0aW9uIHRvIGNhbGwgZm9yIHZhbGlkYXRpb24uXG4gICAqIEBwYXJhbSB2YWxJbnMgTGlzdCBvZiB0ZW5zb3JzIHRvIGJlIGZlZCB0byBgdmFsRmAuXG4gICAqIEBwYXJhbSBzaHVmZmxlIFdoZXRoZXIgdG8gc2h1ZmZsZSB0aGUgZGF0YSBhdCB0aGUgYmVnaW5uaW5nIG9mIGV2ZXJ5XG4gICAqIGVwb2NoLiBEZWZhdWx0IDogdHJ1ZS5cbiAgICogQHBhcmFtIGNhbGxiYWNrTWV0cmljcyBMaXN0IG9mIHN0cmluZ3MsIHRoZSBkaXNwbGF5IG5hbWVzIG9mIHRoZSBtZXRyaWNzXG4gICAqICAgcGFzc2VkIHRvIHRoZSBjYWxsYmFja3MuIFRoZXkgc2hvdWxkIGJlIHRoZSBjb25jYXRlbmF0aW9uIG9mIHRoZVxuICAgKiAgIGRpc3BsYXkgbmFtZXMgb2YgdGhlIG91dHB1dHMgb2YgYGZgIGFuZCB0aGUgbGlzdCBvZiBkaXNwbGF5IG5hbWVzXG4gICAqICAgb2YgdGhlIG91dHB1dHMgb2YgYHZhbEZgLlxuICAgKiBAcGFyYW0gaW5pdGlhbEVwb2NoIEVwb2NoIGF0IHdoaWNoIHRvIHN0YXJ0IHRyYWluaW5nICh1c2VmdWwgZm9yXG4gICAqICAgcmVzdW1pbmcgYSBwcmV2aW91cyB0cmFpbmluZyBydW4pLiBEZWZhdWx0IDogMC5cbiAgICogQHBhcmFtIHN0ZXBzUGVyRXBvY2ggVG90YWwgbnVtYmVyIG9mIHN0ZXBzIChiYXRjaGVzIG9uIHNhbXBsZXMpIGJlZm9yZVxuICAgKiAgIGRlY2xhcmluZyBvbmUgZXBvY2ggZmluaXNoZWQgYW5kIHN0YXJ0aW5nIHRoZSBuZXh0IGVwb2NoLiBJZ25vcmVkIHdpdGhcbiAgICogICB0aGUgZGVmYXVsdCB2YWx1ZSBvZiBgdW5kZWZpbmVkYCBvciBgbnVsbGAuXG4gICAqIEBwYXJhbSB2YWxpZGF0aW9uU3RlcHMgTnVtYmVyIG9mIHN0ZXBzIHRvIHJ1biB2YWxpZGF0aW9uIGZvciAob25seSBpZlxuICAgKiAgIGRvaW5nIHZhbGlkYXRpb24gZnJvbSBkYXRhIHRlbnNvcnMpLiBOb3QgYXBwbGljYWJsZSBmb3IgdGZqcy1sYXllcnMuXG4gICAqIEByZXR1cm5zIEEgYEhpc3RvcnlgIG9iamVjdC5cbiAgICovXG4gIGFzeW5jIGZpdExvb3AoXG4gICAgICBmOiAoZGF0YTogVGVuc29yW10pID0+IFNjYWxhcltdLCBpbnM6IFRlbnNvcltdLCBvdXRMYWJlbHM/OlxuICAgICAgc3RyaW5nW10sIGJhdGNoU2l6ZT86IG51bWJlciwgZXBvY2hzPzogbnVtYmVyLCB2ZXJib3NlPzogbnVtYmVyLFxuICAgICAgY2FsbGJhY2tzPzogQmFzZUNhbGxiYWNrW10sIHZhbEY/OiAoZGF0YTogVGVuc29yW10pID0+IFNjYWxhcltdLCB2YWxJbnM/OlxuICAgICAgVGVuc29yW10sIHNodWZmbGU/OiBib29sZWFufHN0cmluZywgY2FsbGJhY2tNZXRyaWNzPzogc3RyaW5nW10sXG4gICAgICBpbml0aWFsRXBvY2g/OiBudW1iZXIsIHN0ZXBzUGVyRXBvY2g/OiBudW1iZXIsIHZhbGlkYXRpb25TdGVwcz86IG51bWJlcik6XG4gICAgICBQcm9taXNlPEhpc3Rvcnk+IHtcbiAgICBpZiAoYmF0Y2hTaXplID09IG51bGwpIHtcbiAgICAgIGJhdGNoU2l6ZSA9IDMyO1xuICAgIH1cbiAgICBpZiAoZXBvY2hzID09IG51bGwpIHtcbiAgICAgIGVwb2NocyA9IDE7XG4gICAgfVxuICAgIGlmIChzaHVmZmxlID09IG51bGwpIHtcbiAgICAgIHNodWZmbGUgPSB0cnVlO1xuICAgIH1cbiAgICBpZiAoaW5pdGlhbEVwb2NoID09IG51bGwpIHtcbiAgICAgIGluaXRpYWxFcG9jaCA9IDA7XG4gICAgfVxuXG4gICAgLy8gVE9ETyhjYWlzKTogQ2hhbmdlIGNvbnN0IHRvIGxldCBiZWxvdyB3aGVuIGltcGxlbWVudGluZyB2YWxpZGF0aW9uLlxuICAgIGxldCBkb1ZhbGlkYXRpb24gPSBmYWxzZTtcbiAgICBpZiAodmFsRiAhPSBudWxsICYmIHZhbElucyAhPSBudWxsKSB7XG4gICAgICBkb1ZhbGlkYXRpb24gPSB0cnVlO1xuICAgICAgLy8gVE9ETyhjYWlzKTogdmVyYm9zZSBtZXNzYWdlLlxuICAgIH1cbiAgICBpZiAodmFsaWRhdGlvblN0ZXBzICE9IG51bGwpIHtcbiAgICAgIGRvVmFsaWRhdGlvbiA9IHRydWU7XG4gICAgICBpZiAoc3RlcHNQZXJFcG9jaCA9PSBudWxsKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgJ0NhbiBvbmx5IHVzZSBgdmFsaWRhdGlvblN0ZXBzYCB3aGVuIGRvaW5nIHN0ZXAtd2lzZSB0cmFpbmluZywgJyArXG4gICAgICAgICAgICAnaS5lLiwgYHN0ZXBzUGVyRXBvY2hgIG11c3QgYmUgc2V0LicpO1xuICAgICAgfVxuICAgIH1cblxuICAgIGNvbnN0IG51bVRyYWluU2FtcGxlcyA9XG4gICAgICAgIHRoaXMuY2hlY2tOdW1TYW1wbGVzKGlucywgYmF0Y2hTaXplLCBzdGVwc1BlckVwb2NoLCAnc3RlcHNfcGVyX2Vwb2NoJyk7XG4gICAgbGV0IGluZGV4QXJyYXk6IG51bWJlcltdO1xuICAgIGlmIChudW1UcmFpblNhbXBsZXMgIT0gbnVsbCkge1xuICAgICAgaW5kZXhBcnJheSA9IHJhbmdlKDAsIG51bVRyYWluU2FtcGxlcyk7XG4gICAgfVxuXG4gICAgaWYgKHZlcmJvc2UgPT0gbnVsbCkge1xuICAgICAgdmVyYm9zZSA9IDE7XG4gICAgfVxuXG4gICAgY29uc3Qge2NhbGxiYWNrTGlzdCwgaGlzdG9yeX0gPSBjb25maWd1cmVDYWxsYmFja3MoXG4gICAgICAgIGNhbGxiYWNrcywgdmVyYm9zZSwgZXBvY2hzLCBpbml0aWFsRXBvY2gsIG51bVRyYWluU2FtcGxlcyxcbiAgICAgICAgc3RlcHNQZXJFcG9jaCwgYmF0Y2hTaXplLCBkb1ZhbGlkYXRpb24sIGNhbGxiYWNrTWV0cmljcyk7XG4gICAgY2FsbGJhY2tMaXN0LnNldE1vZGVsKHRoaXMpO1xuICAgIHRoaXMuaGlzdG9yeSA9IGhpc3Rvcnk7XG4gICAgYXdhaXQgY2FsbGJhY2tMaXN0Lm9uVHJhaW5CZWdpbigpO1xuICAgIHRoaXMuc3RvcFRyYWluaW5nXyA9IGZhbHNlO1xuICAgIC8vIFRPRE8oY2Fpcyk6IFRha2UgY2FyZSBvZiBjYWxsYmFja3MudmFsaWRhdGlvbl9kYXRhIGFzIGluIFB5S2VyYXMuXG4gICAgLy8gVE9ETyhjYWlzKTogUHJlLWNvbnZlcnQgZmVlZHMgZm9yIHBlcmZvcm1hbmNlIGFzIGluIFB5S2VyYXMuXG5cbiAgICBmb3IgKGxldCBlcG9jaCA9IGluaXRpYWxFcG9jaDsgZXBvY2ggPCBlcG9jaHM7ICsrZXBvY2gpIHtcbiAgICAgIGF3YWl0IGNhbGxiYWNrTGlzdC5vbkVwb2NoQmVnaW4oZXBvY2gpO1xuICAgICAgY29uc3QgZXBvY2hMb2dzOiBVbnJlc29sdmVkTG9ncyA9IHt9O1xuICAgICAgaWYgKHN0ZXBzUGVyRXBvY2ggIT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICAgICdzdGVwc1BlckVwb2NoIG1vZGUgaXMgbm90IGltcGxlbWVudGVkIHlldC4nKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGlmIChzaHVmZmxlID09PSAnYmF0Y2gnKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoJ2JhdGNoIHNodWZmbGluZyBpcyBub3QgaW1wbGVtbmV0ZWQnXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKyAnIHlldCcpO1xuICAgICAgICB9IGVsc2UgaWYgKHNodWZmbGUpIHtcbiAgICAgICAgICB1dGlsLnNodWZmbGUoaW5kZXhBcnJheSk7XG4gICAgICAgIH1cbiAgICAgICAgLy8gQ29udmVydCB0aGUgcG90ZW50aWFsbHkgc2h1ZmZsZWQgaW5kaWNlcyB0byBUZW5zb3IxRCwgdG8gYXZvaWQgdGhlXG4gICAgICAgIC8vIGNvc3Qgb2YgcmVwZWF0ZWQgY3JlYXRpb24gb2YgQXJyYXkxRHMgbGF0ZXIgb24uXG4gICAgICAgIGNvbnN0IGVwb2NoSW5kZXhBcnJheTFEID0gdGVuc29yMWQoaW5kZXhBcnJheSk7XG5cbiAgICAgICAgY29uc3QgYmF0Y2hlcyA9IG1ha2VCYXRjaGVzKG51bVRyYWluU2FtcGxlcywgYmF0Y2hTaXplKTtcbiAgICAgICAgZm9yIChsZXQgYmF0Y2hJbmRleCA9IDA7IGJhdGNoSW5kZXggPCBiYXRjaGVzLmxlbmd0aDsgKytiYXRjaEluZGV4KSB7XG4gICAgICAgICAgY29uc3QgYmF0Y2hMb2dzOiBVbnJlc29sdmVkTG9ncyA9IHt9O1xuICAgICAgICAgIGF3YWl0IGNhbGxiYWNrTGlzdC5vbkJhdGNoQmVnaW4oYmF0Y2hJbmRleCwgYmF0Y2hMb2dzKTtcblxuICAgICAgICAgIHRmYy50aWR5KCgpID0+IHtcbiAgICAgICAgICAgIGNvbnN0IGJhdGNoU3RhcnQgPSBiYXRjaGVzW2JhdGNoSW5kZXhdWzBdO1xuICAgICAgICAgICAgY29uc3QgYmF0Y2hFbmQgPSBiYXRjaGVzW2JhdGNoSW5kZXhdWzFdO1xuICAgICAgICAgICAgY29uc3QgYmF0Y2hJZHMgPSBLLnNsaWNlQWxvbmdGaXJzdEF4aXMoXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBlcG9jaEluZGV4QXJyYXkxRCwgYmF0Y2hTdGFydCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJhdGNoRW5kIC0gYmF0Y2hTdGFydCkgYXMgVGVuc29yMUQ7XG4gICAgICAgICAgICBiYXRjaExvZ3NbJ2JhdGNoJ10gPSBiYXRjaEluZGV4O1xuICAgICAgICAgICAgYmF0Y2hMb2dzWydzaXplJ10gPSBiYXRjaEVuZCAtIGJhdGNoU3RhcnQ7XG5cbiAgICAgICAgICAgIC8vIFRPRE8oY2Fpcyk6IEluIGlucywgdHJhaW4gZmxhZyBjYW4gYmUgYSBudW1iZXIsIGluc3RlYWQgb2YgYW5cbiAgICAgICAgICAgIC8vICAgVGVuc29yPyBEbyB3ZSBuZWVkIHRvIGhhbmRsZSB0aGlzIGluIHRmanMtbGF5ZXJzP1xuICAgICAgICAgICAgY29uc3QgaW5zQmF0Y2ggPSBzbGljZUFycmF5c0J5SW5kaWNlcyhpbnMsIGJhdGNoSWRzKSBhcyBUZW5zb3JbXTtcbiAgICAgICAgICAgIGNvbnN0IG91dHMgPSBmKGluc0JhdGNoKTtcbiAgICAgICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgb3V0TGFiZWxzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgICAgIGNvbnN0IGxhYmVsID0gb3V0TGFiZWxzW2ldO1xuICAgICAgICAgICAgICBjb25zdCBvdXQgPSBvdXRzW2ldO1xuICAgICAgICAgICAgICBiYXRjaExvZ3NbbGFiZWxdID0gb3V0O1xuICAgICAgICAgICAgICB0ZmMua2VlcChvdXQpO1xuICAgICAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBVc2Ugc2NvcGUoKSB0byBhdm9pZCBvd25lcnNoaXAuXG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGlmIChiYXRjaEluZGV4ID09PSBiYXRjaGVzLmxlbmd0aCAtIDEpIHsgIC8vIExhc3QgYmF0Y2guXG4gICAgICAgICAgICAgIGlmIChkb1ZhbGlkYXRpb24pIHtcbiAgICAgICAgICAgICAgICBjb25zdCB2YWxPdXRzID0gdGhpcy50ZXN0TG9vcCh2YWxGLCB2YWxJbnMsIGJhdGNoU2l6ZSk7XG4gICAgICAgICAgICAgICAgLy8gUG9ydGluZyBOb3RlczogSW4gdGZqcy1sYXllcnMsIHZhbE91dHMgaXMgYWx3YXlzIGFuIEFycmF5LlxuICAgICAgICAgICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgb3V0TGFiZWxzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgICAgICAgICBjb25zdCBsYWJlbCA9IG91dExhYmVsc1tpXTtcbiAgICAgICAgICAgICAgICAgIGNvbnN0IG91dCA9IHZhbE91dHNbaV07XG4gICAgICAgICAgICAgICAgICB0ZmMua2VlcChvdXQpO1xuICAgICAgICAgICAgICAgICAgLy8gVE9ETyhjYWlzKTogVXNlIHNjb3BlKCkgdG8gYXZvaWQgb3duZXJzaGlwLlxuICAgICAgICAgICAgICAgICAgZXBvY2hMb2dzWyd2YWxfJyArIGxhYmVsXSA9IG91dDtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9KTtcblxuICAgICAgICAgIGF3YWl0IGNhbGxiYWNrTGlzdC5vbkJhdGNoRW5kKGJhdGNoSW5kZXgsIGJhdGNoTG9ncyk7XG4gICAgICAgICAgZGlzcG9zZVRlbnNvcnNJbkxvZ3MoYmF0Y2hMb2dzKTtcblxuICAgICAgICAgIGlmICh0aGlzLnN0b3BUcmFpbmluZ18pIHtcbiAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgIH1cbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiByZXR1cm4gb3V0cyBhcyBsaXN0IG9mIFRlbnNvci5cbiAgICAgICAgfVxuXG4gICAgICAgIGVwb2NoSW5kZXhBcnJheTFELmRpc3Bvc2UoKTtcbiAgICAgIH1cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IFJ1biB2YWxpZGF0aW9uIGF0IHRoZSBlbmQgb2YgdGhlIGVwb2NoLlxuICAgICAgYXdhaXQgY2FsbGJhY2tMaXN0Lm9uRXBvY2hFbmQoZXBvY2gsIGVwb2NoTG9ncyk7XG4gICAgICBpZiAodGhpcy5zdG9wVHJhaW5pbmdfKSB7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgIH1cbiAgICBhd2FpdCBjYWxsYmFja0xpc3Qub25UcmFpbkVuZCgpO1xuXG4gICAgYXdhaXQgdGhpcy5oaXN0b3J5LnN5bmNEYXRhKCk7XG4gICAgcmV0dXJuIHRoaXMuaGlzdG9yeTtcbiAgfVxuXG4gIC8vIFRPRE8oY2Fpcyk6IEFkZCBjb2RlIHNuaXBwZXQgYmVsb3cgd2hlbiBpdCdzIHBvc3NpYmxlIHRvIGluc3RhbnRpYXRlXG4gIC8vICAgYWN0dWFsIGRhdGFzZXQgb2JqZWN0cy5cbiAgLyoqXG4gICAqIFRyYWlucyB0aGUgbW9kZWwgdXNpbmcgYSBkYXRhc2V0IG9iamVjdC5cbiAgICpcbiAgICogQHBhcmFtIGRhdGFzZXQgQSBkYXRhc2V0IG9iamVjdC4gSXRzIGBpdGVyYXRvcigpYCBtZXRob2QgaXMgZXhwZWN0ZWRcbiAgICogICB0byBnZW5lcmF0ZSBhIGRhdGFzZXQgaXRlcmF0b3Igb2JqZWN0LCB0aGUgYG5leHQoKWAgbWV0aG9kIG9mIHdoaWNoXG4gICAqICAgaXMgZXhwZWN0ZWQgdG8gcHJvZHVjZSBkYXRhIGJhdGNoZXMgZm9yIHRyYWluaW5nLiBUaGUgcmV0dXJuIHZhbHVlXG4gICAqICAgb2YgdGhlIGBuZXh0KClgIGNhbGwgb3VnaHQgdG8gY29udGFpbiBhIGJvb2xlYW4gYGRvbmVgIGZpZWxkIGFuZCBhXG4gICAqICAgYHZhbHVlYCBmaWVsZC4gVGhlIGB2YWx1ZWAgZmllbGQgaXMgZXhwZWN0ZWQgdG8gYmUgYW4gYXJyYXkgb2YgdHdvXG4gICAqICAgYHRmLlRlbnNvcmBzIG9yIGFuIGFycmF5IG9mIHR3byBuZXN0ZWQgYHRmLlRlbnNvcmAgc3RydWN0dXJlcy4gVGhlIGZvcm1lclxuICAgKiAgIGNhc2UgaXMgZm9yIG1vZGVscyB3aXRoIGV4YWN0bHkgb25lIGlucHV0IGFuZCBvbmUgb3V0cHV0IChlLmcuXG4gICAqICAgYSBzZXF1ZW50aWFsIG1vZGVsKS4gVGhlIGxhdHRlciBjYXNlIGlzIGZvciBtb2RlbHMgd2l0aCBtdWx0aXBsZVxuICAgKiAgIGlucHV0cyBhbmQvb3IgbXVsdGlwbGUgb3V0cHV0cy5cbiAgICogICBPZiB0aGUgdHdvIGl0ZW1zIGluIHRoZSBhcnJheSwgdGhlIGZpcnN0IGlzIHRoZSBpbnB1dCBmZWF0dXJlKHMpIGFuZFxuICAgKiAgIHRoZSBzZWNvbmQgaXMgdGhlIG91dHB1dCB0YXJnZXQocykuXG4gICAqIEBwYXJhbSBhcmdzIEEgYE1vZGVsRml0RGF0YXNldEFyZ3NgLCBjb250YWluaW5nIG9wdGlvbmFsIGZpZWxkcy5cbiAgICpcbiAgICogQHJldHVybiBBIGBIaXN0b3J5YCBpbnN0YW5jZS4gSXRzIGBoaXN0b3J5YCBhdHRyaWJ1dGUgY29udGFpbnMgYWxsXG4gICAqICAgaW5mb3JtYXRpb24gY29sbGVjdGVkIGR1cmluZyB0cmFpbmluZy5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFzeW5jIGZpdERhdGFzZXQ8VD4oZGF0YXNldDogRGF0YXNldDxUPiwgYXJnczogTW9kZWxGaXREYXRhc2V0QXJnczxUPik6XG4gICAgICBQcm9taXNlPEhpc3Rvcnk+IHtcbiAgICByZXR1cm4gZml0RGF0YXNldCh0aGlzLCBkYXRhc2V0LCBhcmdzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSdW5zIGEgc2luZ2xlIGdyYWRpZW50IHVwZGF0ZSBvbiBhIHNpbmdsZSBiYXRjaCBvZiBkYXRhLlxuICAgKlxuICAgKiBUaGlzIG1ldGhvZCBkaWZmZXJzIGZyb20gYGZpdCgpYCBhbmQgYGZpdERhdGFzZXQoKWAgaW4gdGhlIGZvbGxvd2luZ1xuICAgKiByZWdhcmRzOlxuICAgKiAgIC0gSXQgb3BlcmF0ZXMgb24gZXhhY3RseSBvbmUgYmF0Y2ggb2YgZGF0YS5cbiAgICogICAtIEl0IHJldHVybnMgb25seSB0aGUgbG9zcyBhbmQgbWV0cmljIHZhbHVlcywgaW5zdGVhZCBvZlxuICAgKiAgICAgcmV0dXJuaW5nIHRoZSBiYXRjaC1ieS1iYXRjaCBsb3NzIGFuZCBtZXRyaWMgdmFsdWVzLlxuICAgKiAgIC0gSXQgZG9lc24ndCBzdXBwb3J0IGZpbmUtZ3JhaW5lZCBvcHRpb25zIHN1Y2ggYXMgdmVyYm9zaXR5IGFuZFxuICAgKiAgICAgY2FsbGJhY2tzLlxuICAgKlxuICAgKiBAcGFyYW0geCBJbnB1dCBkYXRhLiBJdCBjb3VsZCBiZSBvbmUgb2YgdGhlIGZvbGxvd2luZzpcbiAgICogICAtIEEgYHRmLlRlbnNvcmAsIG9yIGFuIEFycmF5IG9mIGB0Zi5UZW5zb3JgcyAoaW4gY2FzZSB0aGUgbW9kZWwgaGFzXG4gICAqICAgICBtdWx0aXBsZSBpbnB1dHMpLlxuICAgKiAgIC0gQW4gT2JqZWN0IG1hcHBpbmcgaW5wdXQgbmFtZXMgdG8gY29ycmVzcG9uZGluZyBgdGYuVGVuc29yYCAoaWYgdGhlXG4gICAqICAgICBtb2RlbCBoYXMgbmFtZWQgaW5wdXRzKS5cbiAgICogQHBhcmFtIHkgVGFyZ2V0IGRhdGEuIEl0IGNvdWxkIGJlIGVpdGhlciBhIGB0Zi5UZW5zb3JgIG9yIG11bHRpcGxlXG4gICAqICAgYHRmLlRlbnNvcmBzLiBJdCBzaG91bGQgYmUgY29uc2lzdGVudCB3aXRoIGB4YC5cbiAgICogQHJldHVybnMgVHJhaW5pbmcgbG9zcyBvciBsb3NzZXMgKGluIGNhc2UgdGhlIG1vZGVsIGhhc1xuICAgKiAgIG11bHRpcGxlIG91dHB1dHMpLCBhbG9uZyB3aXRoIG1ldHJpY3MgKGlmIGFueSksIGFzIG51bWJlcnMuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBhc3luYyB0cmFpbk9uQmF0Y2goXG4gICAgICB4OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgICB5OiBUZW5zb3J8VGVuc29yW118XG4gICAgICB7W2lucHV0TmFtZTogc3RyaW5nXTogVGVuc29yfSk6IFByb21pc2U8bnVtYmVyfG51bWJlcltdPiB7XG4gICAgLy8gVE9ETyhjYWlzKTogU3VwcG9ydCBzYW1wbGVXZWlnaHQgYW5kIGNsYXNzV2VpZ2h0LlxuICAgIC8vIFRPRE8oY2Fpcyk6IFN1cHBvcnQgRGF0YXNldCBvYmplY3RzLlxuICAgIGNvbnN0IHN0YW5kYXJkaXplT3V0ID0gYXdhaXQgdGhpcy5zdGFuZGFyZGl6ZVVzZXJEYXRhKHgsIHkpO1xuICAgIGNvbnN0IGlucHV0cyA9IHN0YW5kYXJkaXplT3V0WzBdO1xuICAgIGNvbnN0IHRhcmdldHMgPSBzdGFuZGFyZGl6ZU91dFsxXTtcbiAgICBjb25zdCB0cmFpbkZ1bmN0aW9uID0gdGhpcy5tYWtlVHJhaW5GdW5jdGlvbigpO1xuICAgIGNvbnN0IGxvc3NlcyA9IHRyYWluRnVuY3Rpb24oaW5wdXRzLmNvbmNhdCh0YXJnZXRzKSk7XG4gICAgY29uc3QgbG9zc1ZhbHVlczogbnVtYmVyW10gPSBbXTtcbiAgICBmb3IgKGNvbnN0IGxvc3Mgb2YgbG9zc2VzKSB7XG4gICAgICBjb25zdCB2ID0gYXdhaXQgbG9zcy5kYXRhKCk7XG4gICAgICBsb3NzVmFsdWVzLnB1c2godlswXSk7XG4gICAgfVxuICAgIHRmYy5kaXNwb3NlKGxvc3Nlcyk7XG4gICAgZGlzcG9zZU5ld1RlbnNvcnMoc3RhbmRhcmRpemVPdXRbMF0sIHgpO1xuICAgIGRpc3Bvc2VOZXdUZW5zb3JzKHN0YW5kYXJkaXplT3V0WzFdLCB5KTtcbiAgICByZXR1cm4gc2luZ2xldG9uT3JBcnJheShsb3NzVmFsdWVzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBFeHRyYWN0IHdlaWdodCB2YWx1ZXMgb2YgdGhlIG1vZGVsLlxuICAgKlxuICAgKiBAcGFyYW0gY29uZmlnOiBBbiBpbnN0YW5jZSBvZiBgaW8uU2F2ZUNvbmZpZ2AsIHdoaWNoIHNwZWNpZmllc1xuICAgKiBtb2RlbC1zYXZpbmcgb3B0aW9ucyBzdWNoIGFzIHdoZXRoZXIgb25seSB0cmFpbmFibGUgd2VpZ2h0cyBhcmUgdG8gYmVcbiAgICogc2F2ZWQuXG4gICAqIEByZXR1cm5zIEEgYE5hbWVkVGVuc29yTWFwYCBtYXBwaW5nIG9yaWdpbmFsIHdlaWdodCBuYW1lcyAoaS5lLixcbiAgICogICBub24tdW5pcXVlaWZpZWQgd2VpZ2h0IG5hbWVzKSB0byB0aGVpciB2YWx1ZXMuXG4gICAqL1xuICBwcm90ZWN0ZWQgZ2V0TmFtZWRXZWlnaHRzKGNvbmZpZz86IGlvLlNhdmVDb25maWcpOiBOYW1lZFRlbnNvcltdIHtcbiAgICBjb25zdCBuYW1lZFdlaWdodHM6IE5hbWVkVGVuc29yW10gPSBbXTtcblxuICAgIGNvbnN0IHRyYWluYWJsZU9ubHkgPSBjb25maWcgIT0gbnVsbCAmJiBjb25maWcudHJhaW5hYmxlT25seTtcbiAgICBjb25zdCB3ZWlnaHRzID0gdHJhaW5hYmxlT25seSA/IHRoaXMudHJhaW5hYmxlV2VpZ2h0cyA6IHRoaXMud2VpZ2h0cztcbiAgICBjb25zdCB3ZWlnaHRWYWx1ZXMgPSB0aGlzLmdldFdlaWdodHModHJhaW5hYmxlT25seSk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB3ZWlnaHRzLmxlbmd0aDsgKytpKSB7XG4gICAgICBpZiAodHJhaW5hYmxlT25seSAmJiAhd2VpZ2h0c1tpXS50cmFpbmFibGUpIHtcbiAgICAgICAgLy8gT3B0aW9uYWxseSBza2lwIG5vbi10cmFpbmFibGUgd2VpZ2h0cy5cbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBuYW1lZFdlaWdodHMucHVzaChcbiAgICAgICAgICB7bmFtZTogd2VpZ2h0c1tpXS5vcmlnaW5hbE5hbWUsIHRlbnNvcjogd2VpZ2h0VmFsdWVzW2ldfSk7XG4gICAgfVxuICAgIHJldHVybiBuYW1lZFdlaWdodHM7XG4gIH1cblxuICAvKipcbiAgICogU2V0dGVyIHVzZWQgZm9yIGZvcmNlIHN0b3BwaW5nIG9mIExheWVyc01vZGVsLmZpdCgpIChpLmUuLCB0cmFpbmluZykuXG4gICAqXG4gICAqIEV4YW1wbGU6XG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGlucHV0ID0gdGYuaW5wdXQoe3NoYXBlOiBbMTBdfSk7XG4gICAqIGNvbnN0IG91dHB1dCA9IHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDF9KS5hcHBseShpbnB1dCk7XG4gICAqIGNvbnN0IG1vZGVsID0gdGYubW9kZWwoe2lucHV0czogW2lucHV0XSwgb3V0cHV0czogW291dHB1dF19KTtcbiAgICogbW9kZWwuY29tcGlsZSh7bG9zczogJ21lYW5TcXVhcmVkRXJyb3InLCBvcHRpbWl6ZXI6ICdzZ2QnfSk7XG4gICAqIGNvbnN0IHhzID0gdGYub25lcyhbOCwgMTBdKTtcbiAgICogY29uc3QgeXMgPSB0Zi56ZXJvcyhbOCwgMV0pO1xuICAgKlxuICAgKiBjb25zdCBoaXN0b3J5ID0gYXdhaXQgbW9kZWwuZml0KHhzLCB5cywge1xuICAgKiAgIGVwb2NoczogMTAsXG4gICAqICAgY2FsbGJhY2tzOiB7XG4gICAqICAgICBvbkVwb2NoRW5kOiBhc3luYyAoZXBvY2gsIGxvZ3MpID0+IHtcbiAgICogICAgICAgaWYgKGVwb2NoID09PSAyKSB7XG4gICAqICAgICAgICAgbW9kZWwuc3RvcFRyYWluaW5nID0gdHJ1ZTtcbiAgICogICAgICAgfVxuICAgKiAgICAgfVxuICAgKiAgIH1cbiAgICogfSk7XG4gICAqXG4gICAqIC8vIFRoZXJlIHNob3VsZCBiZSBvbmx5IDMgdmFsdWVzIGluIHRoZSBsb3NzIGFycmF5LCBpbnN0ZWFkIG9mIDEwXG4gICAqIHZhbHVlcyxcbiAgICogLy8gZHVlIHRvIHRoZSBzdG9wcGluZyBhZnRlciAzIGVwb2Nocy5cbiAgICogY29uc29sZS5sb2coaGlzdG9yeS5oaXN0b3J5Lmxvc3MpO1xuICAgKiBgYGBcbiAgICovXG4gIHNldCBzdG9wVHJhaW5pbmcoc3RvcDogYm9vbGVhbikge1xuICAgIHRoaXMuc3RvcFRyYWluaW5nXyA9IHN0b3A7XG4gIH1cblxuICBnZXQgc3RvcFRyYWluaW5nKCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiB0aGlzLnN0b3BUcmFpbmluZ187XG4gIH1cblxuICBnZXQgb3B0aW1pemVyKCk6IE9wdGltaXplciB7XG4gICAgcmV0dXJuIHRoaXMub3B0aW1pemVyXztcbiAgfVxuXG4gIHNldCBvcHRpbWl6ZXIob3B0aW1pemVyOiBPcHRpbWl6ZXIpIHtcbiAgICBpZiAodGhpcy5vcHRpbWl6ZXJfICE9PSBvcHRpbWl6ZXIpIHtcbiAgICAgIHRoaXMub3B0aW1pemVyXyA9IG9wdGltaXplcjtcbiAgICAgIHRoaXMuaXNPcHRpbWl6ZXJPd25lZCA9IGZhbHNlO1xuICAgIH1cbiAgfVxuXG4gIG92ZXJyaWRlIGRpc3Bvc2UoKTogRGlzcG9zZVJlc3VsdCB7XG4gICAgY29uc3QgcmVzdWx0ID0gc3VwZXIuZGlzcG9zZSgpO1xuICAgIGlmIChyZXN1bHQucmVmQ291bnRBZnRlckRpc3Bvc2UgPT09IDAgJiYgdGhpcy5vcHRpbWl6ZXIgIT0gbnVsbCAmJlxuICAgICAgICB0aGlzLmlzT3B0aW1pemVyT3duZWQpIHtcbiAgICAgIGNvbnN0IG51bVRlbnNvcnNCZWZvcmVPcHRtaXplckRpc3Bvc2FsID0gdGZjLm1lbW9yeSgpLm51bVRlbnNvcnM7XG4gICAgICB0aGlzLm9wdGltaXplcl8uZGlzcG9zZSgpO1xuICAgICAgcmVzdWx0Lm51bURpc3Bvc2VkVmFyaWFibGVzICs9XG4gICAgICAgICAgbnVtVGVuc29yc0JlZm9yZU9wdG1pemVyRGlzcG9zYWwgLSB0ZmMubWVtb3J5KCkubnVtVGVuc29ycztcbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByaXZhdGUgZ2V0TG9zc0lkZW50aWZpZXJzKCk6IExvc3NJZGVudGlmaWVyfExvc3NJZGVudGlmaWVyW118XG4gICAgICB7W291dHB1dE5hbWU6IHN0cmluZ106IExvc3NJZGVudGlmaWVyfSB7XG4gICAgbGV0IGxvc3NOYW1lczogTG9zc0lkZW50aWZpZXJ8TG9zc0lkZW50aWZpZXJbXXxcbiAgICAgICAge1tvdXRwdXROYW1lOiBzdHJpbmddOiBMb3NzSWRlbnRpZmllcn07XG4gICAgaWYgKHR5cGVvZiB0aGlzLmxvc3MgPT09ICdzdHJpbmcnKSB7XG4gICAgICBsb3NzTmFtZXMgPSB0b1NuYWtlQ2FzZSh0aGlzLmxvc3MpIGFzIExvc3NJZGVudGlmaWVyO1xuICAgIH0gZWxzZSBpZiAoQXJyYXkuaXNBcnJheSh0aGlzLmxvc3MpKSB7XG4gICAgICBmb3IgKGNvbnN0IGxvc3Mgb2YgdGhpcy5sb3NzKSB7XG4gICAgICAgIGlmICh0eXBlb2YgbG9zcyAhPT0gJ3N0cmluZycpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1NlcmlhbGl6YXRpb24gb2Ygbm9uLXN0cmluZyBsb3NzIGlzIG5vdCBzdXBwb3J0ZWQuJyk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGxvc3NOYW1lcyA9ICh0aGlzLmxvc3MgYXMgc3RyaW5nW10pLm1hcChuYW1lID0+IHRvU25ha2VDYXNlKG5hbWUpKSBhc1xuICAgICAgICAgIExvc3NJZGVudGlmaWVyW107XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IG91dHB1dE5hbWVzID0gT2JqZWN0LmtleXModGhpcy5sb3NzKTtcbiAgICAgIGxvc3NOYW1lcyA9IHt9IGFzIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogTG9zc0lkZW50aWZpZXJ9O1xuICAgICAgY29uc3QgbG9zc2VzID1cbiAgICAgICAgICB0aGlzLmxvc3MgYXMge1tvdXRwdXROYW1lOiBzdHJpbmddOiBMb3NzT3JNZXRyaWNGbiB8IHN0cmluZ307XG4gICAgICBmb3IgKGNvbnN0IG91dHB1dE5hbWUgb2Ygb3V0cHV0TmFtZXMpIHtcbiAgICAgICAgaWYgKHR5cGVvZiBsb3NzZXNbb3V0cHV0TmFtZV0gPT09ICdzdHJpbmcnKSB7XG4gICAgICAgICAgbG9zc05hbWVzW291dHB1dE5hbWVdID1cbiAgICAgICAgICAgICAgdG9TbmFrZUNhc2UobG9zc2VzW291dHB1dE5hbWVdIGFzIHN0cmluZykgYXMgTG9zc0lkZW50aWZpZXI7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdTZXJpYWxpemF0aW9uIG9mIG5vbi1zdHJpbmcgbG9zcyBpcyBub3Qgc3VwcG9ydGVkLicpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBsb3NzTmFtZXM7XG4gIH1cblxuICBwcml2YXRlIGdldE1ldHJpY0lkZW50aWZpZXJzKCk6IE1ldHJpY3NJZGVudGlmaWVyW118XG4gICAgICB7W2tleTogc3RyaW5nXTogTWV0cmljc0lkZW50aWZpZXJ9IHtcbiAgICBpZiAodHlwZW9mIHRoaXMubWV0cmljcyA9PT0gJ3N0cmluZycgfHxcbiAgICAgICAgdHlwZW9mIHRoaXMubWV0cmljcyA9PT0gJ2Z1bmN0aW9uJykge1xuICAgICAgcmV0dXJuIFt0b1NuYWtlQ2FzZShNZXRyaWNzLmdldExvc3NPck1ldHJpY05hbWUodGhpcy5tZXRyaWNzKSldO1xuICAgIH0gZWxzZSBpZiAoQXJyYXkuaXNBcnJheSh0aGlzLm1ldHJpY3MpKSB7XG4gICAgICByZXR1cm4gdGhpcy5tZXRyaWNzLm1hcChcbiAgICAgICAgICBtZXRyaWMgPT4gdG9TbmFrZUNhc2UoTWV0cmljcy5nZXRMb3NzT3JNZXRyaWNOYW1lKG1ldHJpYykpKTtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgbWV0cmljc0lkZW50aWZpZXJzOiB7W2tleTogc3RyaW5nXTogTWV0cmljc0lkZW50aWZpZXJ9ID0ge307XG4gICAgICBmb3IgKGNvbnN0IGtleSBpbiB0aGlzLm1ldHJpY3MpIHtcbiAgICAgICAgbWV0cmljc0lkZW50aWZpZXJzW2tleV0gPVxuICAgICAgICAgICAgdG9TbmFrZUNhc2UoTWV0cmljcy5nZXRMb3NzT3JNZXRyaWNOYW1lKHRoaXMubWV0cmljc1trZXldKSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gbWV0cmljc0lkZW50aWZpZXJzO1xuICAgIH1cbiAgfVxuXG4gIHByb3RlY3RlZCBnZXRUcmFpbmluZ0NvbmZpZygpOiBUcmFpbmluZ0NvbmZpZyB7XG4gICAgcmV0dXJuIHtcbiAgICAgIGxvc3M6IHRoaXMuZ2V0TG9zc0lkZW50aWZpZXJzKCksXG4gICAgICBtZXRyaWNzOiB0aGlzLmdldE1ldHJpY0lkZW50aWZpZXJzKCksXG4gICAgICBvcHRpbWl6ZXJfY29uZmlnOiB7XG4gICAgICAgIGNsYXNzX25hbWU6IHRoaXMub3B0aW1pemVyLmdldENsYXNzTmFtZSgpLFxuICAgICAgICBjb25maWc6IHRoaXMub3B0aW1pemVyLmdldENvbmZpZygpXG4gICAgICB9IGFzIE9wdGltaXplclNlcmlhbGl6YXRpb25cbiAgICB9O1xuICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCB3ZWlnaHRfbWV0cmljcyB3aGVuIHRoZXkgYXJlIHN1cHBvcnRlZC5cbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgc2FtcGxlX3dlaWdodF9tb2RlIHdoZW4gaXQncyBzdXBwb3J0ZWQuXG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIGxvc3Nfd2VpZ2h0cyB3aGVuIGl0J3Mgc3VwcG9ydGVkLlxuICB9XG5cbiAgbG9hZFRyYWluaW5nQ29uZmlnKHRyYWluaW5nQ29uZmlnOiBUcmFpbmluZ0NvbmZpZykge1xuICAgIGlmICh0cmFpbmluZ0NvbmZpZy53ZWlnaHRlZF9tZXRyaWNzICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignTG9hZGluZyB3ZWlnaHRfbWV0cmljcyBpcyBub3Qgc3VwcG9ydGVkIHlldC4nKTtcbiAgICB9XG4gICAgaWYgKHRyYWluaW5nQ29uZmlnLmxvc3Nfd2VpZ2h0cyAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0xvYWRpbmcgbG9zc193ZWlnaHRzIGlzIG5vdCBzdXBwb3J0ZWQgeWV0LicpO1xuICAgIH1cbiAgICBpZiAodHJhaW5pbmdDb25maWcuc2FtcGxlX3dlaWdodF9tb2RlICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignTG9hZGluZyBzYW1wbGVfd2VpZ2h0X21vZGUgaXMgbm90IHN1cHBvcnRlZCB5ZXQuJyk7XG4gICAgfVxuXG4gICAgY29uc3QgdHNDb25maWcgPSBjb252ZXJ0UHl0aG9uaWNUb1RzKHRyYWluaW5nQ29uZmlnLm9wdGltaXplcl9jb25maWcpIGFzXG4gICAgICAgIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdDtcbiAgICBjb25zdCBvcHRpbWl6ZXIgPSBkZXNlcmlhbGl6ZSh0c0NvbmZpZykgYXMgT3B0aW1pemVyO1xuXG4gICAgbGV0IGxvc3M7XG4gICAgaWYgKHR5cGVvZiB0cmFpbmluZ0NvbmZpZy5sb3NzID09PSAnc3RyaW5nJykge1xuICAgICAgbG9zcyA9IHRvQ2FtZWxDYXNlKHRyYWluaW5nQ29uZmlnLmxvc3MpO1xuICAgIH0gZWxzZSBpZiAoQXJyYXkuaXNBcnJheSh0cmFpbmluZ0NvbmZpZy5sb3NzKSkge1xuICAgICAgbG9zcyA9IHRyYWluaW5nQ29uZmlnLmxvc3MubWFwKGxvc3NFbnRyeSA9PiB0b0NhbWVsQ2FzZShsb3NzRW50cnkpKTtcbiAgICB9IGVsc2UgaWYgKHRyYWluaW5nQ29uZmlnLmxvc3MgIT0gbnVsbCkge1xuICAgICAgbG9zcyA9IHt9IGFzIHtbb3V0cHV0TmFtZTogc3RyaW5nXTogTG9zc0lkZW50aWZpZXJ9O1xuICAgICAgZm9yIChjb25zdCBrZXkgaW4gdHJhaW5pbmdDb25maWcubG9zcykge1xuICAgICAgICBsb3NzW2tleV0gPSB0b0NhbWVsQ2FzZSh0cmFpbmluZ0NvbmZpZy5sb3NzW2tleV0pIGFzIExvc3NJZGVudGlmaWVyO1xuICAgICAgfVxuICAgIH1cblxuICAgIGxldCBtZXRyaWNzO1xuICAgIGlmIChBcnJheS5pc0FycmF5KHRyYWluaW5nQ29uZmlnLm1ldHJpY3MpKSB7XG4gICAgICBtZXRyaWNzID0gdHJhaW5pbmdDb25maWcubWV0cmljcy5tYXAobWV0cmljID0+IHRvQ2FtZWxDYXNlKG1ldHJpYykpO1xuICAgIH0gZWxzZSBpZiAodHJhaW5pbmdDb25maWcubWV0cmljcyAhPSBudWxsKSB7XG4gICAgICBtZXRyaWNzID0ge30gYXMge1tvdXRwdXROYW1lOiBzdHJpbmddOiBNZXRyaWNzSWRlbnRpZmllcn07XG4gICAgICBmb3IgKGNvbnN0IGtleSBpbiB0cmFpbmluZ0NvbmZpZy5tZXRyaWNzKSB7XG4gICAgICAgIG1ldHJpY3Nba2V5XSA9IHRvQ2FtZWxDYXNlKHRyYWluaW5nQ29uZmlnLm1ldHJpY3Nba2V5XSk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdGhpcy5jb21waWxlKHtsb3NzLCBtZXRyaWNzLCBvcHRpbWl6ZXJ9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTYXZlIHRoZSBjb25maWd1cmF0aW9uIGFuZC9vciB3ZWlnaHRzIG9mIHRoZSBMYXllcnNNb2RlbC5cbiAgICpcbiAgICogQW4gYElPSGFuZGxlcmAgaXMgYW4gb2JqZWN0IHRoYXQgaGFzIGEgYHNhdmVgIG1ldGhvZCBvZiB0aGUgcHJvcGVyXG4gICAqIHNpZ25hdHVyZSBkZWZpbmVkLiBUaGUgYHNhdmVgIG1ldGhvZCBtYW5hZ2VzIHRoZSBzdG9yaW5nIG9yXG4gICAqIHRyYW5zbWlzc2lvbiBvZiBzZXJpYWxpemVkIGRhdGEgKFwiYXJ0aWZhY3RzXCIpIHRoYXQgcmVwcmVzZW50IHRoZVxuICAgKiBtb2RlbCdzIHRvcG9sb2d5IGFuZCB3ZWlnaHRzIG9udG8gb3IgdmlhIGEgc3BlY2lmaWMgbWVkaXVtLCBzdWNoIGFzXG4gICAqIGZpbGUgZG93bmxvYWRzLCBsb2NhbCBzdG9yYWdlLCBJbmRleGVkREIgaW4gdGhlIHdlYiBicm93c2VyIGFuZCBIVFRQXG4gICAqIHJlcXVlc3RzIHRvIGEgc2VydmVyLiBUZW5zb3JGbG93LmpzIHByb3ZpZGVzIGBJT0hhbmRsZXJgXG4gICAqIGltcGxlbWVudGF0aW9ucyBmb3IgYSBudW1iZXIgb2YgZnJlcXVlbnRseSB1c2VkIHNhdmluZyBtZWRpdW1zLCBzdWNoIGFzXG4gICAqIGB0Zi5pby5icm93c2VyRG93bmxvYWRzYCBhbmQgYHRmLmlvLmJyb3dzZXJMb2NhbFN0b3JhZ2VgLiBTZWUgYHRmLmlvYFxuICAgKiBmb3IgbW9yZSBkZXRhaWxzLlxuICAgKlxuICAgKiBUaGlzIG1ldGhvZCBhbHNvIGFsbG93cyB5b3UgdG8gcmVmZXIgdG8gY2VydGFpbiB0eXBlcyBvZiBgSU9IYW5kbGVyYHNcbiAgICogYXMgVVJMLWxpa2Ugc3RyaW5nIHNob3J0Y3V0cywgc3VjaCBhcyAnbG9jYWxzdG9yYWdlOi8vJyBhbmRcbiAgICogJ2luZGV4ZWRkYjovLycuXG4gICAqXG4gICAqIEV4YW1wbGUgMTogU2F2ZSBgbW9kZWxgJ3MgdG9wb2xvZ3kgYW5kIHdlaWdodHMgdG8gYnJvd3NlciBbbG9jYWxcbiAgICogc3RvcmFnZV0oaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL1dpbmRvdy9sb2NhbFN0b3JhZ2UpO1xuICAgKiB0aGVuIGxvYWQgaXQgYmFjay5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKFxuICAgKiAgICAge2xheWVyczogW3RmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFszXX0pXX0pO1xuICAgKiBjb25zb2xlLmxvZygnUHJlZGljdGlvbiBmcm9tIG9yaWdpbmFsIG1vZGVsOicpO1xuICAgKiBtb2RlbC5wcmVkaWN0KHRmLm9uZXMoWzEsIDNdKSkucHJpbnQoKTtcbiAgICpcbiAgICogY29uc3Qgc2F2ZVJlc3VsdHMgPSBhd2FpdCBtb2RlbC5zYXZlKCdsb2NhbHN0b3JhZ2U6Ly9teS1tb2RlbC0xJyk7XG4gICAqXG4gICAqIGNvbnN0IGxvYWRlZE1vZGVsID0gYXdhaXQgdGYubG9hZExheWVyc01vZGVsKCdsb2NhbHN0b3JhZ2U6Ly9teS1tb2RlbC0xJyk7XG4gICAqIGNvbnNvbGUubG9nKCdQcmVkaWN0aW9uIGZyb20gbG9hZGVkIG1vZGVsOicpO1xuICAgKiBsb2FkZWRNb2RlbC5wcmVkaWN0KHRmLm9uZXMoWzEsIDNdKSkucHJpbnQoKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEV4YW1wbGUgMi4gU2F2aW5nIGBtb2RlbGAncyB0b3BvbG9neSBhbmQgd2VpZ2h0cyB0byBicm93c2VyXG4gICAqIFtJbmRleGVkREJdKGh0dHBzOi8vZGV2ZWxvcGVyLm1vemlsbGEub3JnL2VuLVVTL2RvY3MvV2ViL0FQSS9JbmRleGVkREJfQVBJKTtcbiAgICogdGhlbiBsb2FkIGl0IGJhY2suXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbChcbiAgICogICAgIHtsYXllcnM6IFt0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbM119KV19KTtcbiAgICogY29uc29sZS5sb2coJ1ByZWRpY3Rpb24gZnJvbSBvcmlnaW5hbCBtb2RlbDonKTtcbiAgICogbW9kZWwucHJlZGljdCh0Zi5vbmVzKFsxLCAzXSkpLnByaW50KCk7XG4gICAqXG4gICAqIGNvbnN0IHNhdmVSZXN1bHRzID0gYXdhaXQgbW9kZWwuc2F2ZSgnaW5kZXhlZGRiOi8vbXktbW9kZWwtMScpO1xuICAgKlxuICAgKiBjb25zdCBsb2FkZWRNb2RlbCA9IGF3YWl0IHRmLmxvYWRMYXllcnNNb2RlbCgnaW5kZXhlZGRiOi8vbXktbW9kZWwtMScpO1xuICAgKiBjb25zb2xlLmxvZygnUHJlZGljdGlvbiBmcm9tIGxvYWRlZCBtb2RlbDonKTtcbiAgICogbG9hZGVkTW9kZWwucHJlZGljdCh0Zi5vbmVzKFsxLCAzXSkpLnByaW50KCk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBFeGFtcGxlIDMuIFNhdmluZyBgbW9kZWxgJ3MgdG9wb2xvZ3kgYW5kIHdlaWdodHMgYXMgdHdvIGZpbGVzXG4gICAqIChgbXktbW9kZWwtMS5qc29uYCBhbmQgYG15LW1vZGVsLTEud2VpZ2h0cy5iaW5gKSBkb3dubG9hZGVkIGZyb21cbiAgICogYnJvd3Nlci5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKFxuICAgKiAgICAge2xheWVyczogW3RmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFszXX0pXX0pO1xuICAgKiBjb25zdCBzYXZlUmVzdWx0cyA9IGF3YWl0IG1vZGVsLnNhdmUoJ2Rvd25sb2FkczovL215LW1vZGVsLTEnKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEV4YW1wbGUgNC4gU2VuZCAgYG1vZGVsYCdzIHRvcG9sb2d5IGFuZCB3ZWlnaHRzIHRvIGFuIEhUVFAgc2VydmVyLlxuICAgKiBTZWUgdGhlIGRvY3VtZW50YXRpb24gb2YgYHRmLmlvLmh0dHBgIGZvciBtb3JlIGRldGFpbHNcbiAgICogaW5jbHVkaW5nIHNwZWNpZnlpbmcgcmVxdWVzdCBwYXJhbWV0ZXJzIGFuZCBpbXBsZW1lbnRhdGlvbiBvZiB0aGVcbiAgICogc2VydmVyLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoXG4gICAqICAgICB7bGF5ZXJzOiBbdGYubGF5ZXJzLmRlbnNlKHt1bml0czogMSwgaW5wdXRTaGFwZTogWzNdfSldfSk7XG4gICAqIGNvbnN0IHNhdmVSZXN1bHRzID0gYXdhaXQgbW9kZWwuc2F2ZSgnaHR0cDovL215LXNlcnZlci9tb2RlbC91cGxvYWQnKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSBoYW5kbGVyT3JVUkwgQW4gaW5zdGFuY2Ugb2YgYElPSGFuZGxlcmAgb3IgYSBVUkwtbGlrZSxcbiAgICogc2NoZW1lLWJhc2VkIHN0cmluZyBzaG9ydGN1dCBmb3IgYElPSGFuZGxlcmAuXG4gICAqIEBwYXJhbSBjb25maWcgT3B0aW9ucyBmb3Igc2F2aW5nIHRoZSBtb2RlbC5cbiAgICogQHJldHVybnMgQSBgUHJvbWlzZWAgb2YgYFNhdmVSZXN1bHRgLCB3aGljaCBzdW1tYXJpemVzIHRoZSByZXN1bHQgb2ZcbiAgICogdGhlIHNhdmluZywgc3VjaCBhcyBieXRlIHNpemVzIG9mIHRoZSBzYXZlZCBhcnRpZmFjdHMgZm9yIHRoZSBtb2RlbCdzXG4gICAqICAgdG9wb2xvZ3kgYW5kIHdlaWdodCB2YWx1ZXMuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3NlcycsIGlnbm9yZUNJOiB0cnVlfVxuICAgKi9cbiAgYXN5bmMgc2F2ZShoYW5kbGVyT3JVUkw6IGlvLklPSGFuZGxlcnxzdHJpbmcsIGNvbmZpZz86IGlvLlNhdmVDb25maWcpOlxuICAgICAgUHJvbWlzZTxpby5TYXZlUmVzdWx0PiB7XG4gICAgaWYgKHR5cGVvZiBoYW5kbGVyT3JVUkwgPT09ICdzdHJpbmcnKSB7XG4gICAgICBjb25zdCBoYW5kbGVycyA9IGlvLmdldFNhdmVIYW5kbGVycyhoYW5kbGVyT3JVUkwpO1xuICAgICAgaWYgKGhhbmRsZXJzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBDYW5ub3QgZmluZCBhbnkgc2F2ZSBoYW5kbGVycyBmb3IgVVJMICcke2hhbmRsZXJPclVSTH0nYCk7XG4gICAgICB9IGVsc2UgaWYgKGhhbmRsZXJzLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgRm91bmQgbW9yZSB0aGFuIG9uZSAoJHtoYW5kbGVycy5sZW5ndGh9KSBzYXZlIGhhbmRsZXJzIGZvciBgICtcbiAgICAgICAgICAgIGBVUkwgJyR7aGFuZGxlck9yVVJMfSdgKTtcbiAgICAgIH1cbiAgICAgIGhhbmRsZXJPclVSTCA9IGhhbmRsZXJzWzBdO1xuICAgIH1cbiAgICBpZiAoaGFuZGxlck9yVVJMLnNhdmUgPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ0xheWVyc01vZGVsLnNhdmUoKSBjYW5ub3QgcHJvY2VlZCBiZWNhdXNlIHRoZSBJT0hhbmRsZXIgJyArXG4gICAgICAgICAgJ3Byb3ZpZGVkIGRvZXMgbm90IGhhdmUgdGhlIGBzYXZlYCBhdHRyaWJ1dGUgZGVmaW5lZC4nKTtcbiAgICB9XG5cbiAgICBjb25zdCB3ZWlnaHREYXRhQW5kU3BlY3MgPVxuICAgICAgICBhd2FpdCBpby5lbmNvZGVXZWlnaHRzKHRoaXMuZ2V0TmFtZWRXZWlnaHRzKGNvbmZpZykpO1xuXG4gICAgY29uc3QgcmV0dXJuU3RyaW5nID0gZmFsc2U7XG4gICAgY29uc3QgdW51c2VkQXJnOiB7fSA9IG51bGw7XG4gICAgY29uc3QgbW9kZWxDb25maWcgPSB0aGlzLnRvSlNPTih1bnVzZWRBcmcsIHJldHVyblN0cmluZyk7XG4gICAgY29uc3QgbW9kZWxBcnRpZmFjdHM6IGlvLk1vZGVsQXJ0aWZhY3RzID0ge1xuICAgICAgbW9kZWxUb3BvbG9neTogbW9kZWxDb25maWcsXG4gICAgICBmb3JtYXQ6IExBWUVSU19NT0RFTF9GT1JNQVRfTkFNRSxcbiAgICAgIGdlbmVyYXRlZEJ5OiBgVGVuc29yRmxvdy5qcyB0ZmpzLWxheWVycyB2JHt2ZXJzaW9ufWAsXG4gICAgICBjb252ZXJ0ZWRCeTogbnVsbCxcbiAgICB9O1xuXG4gICAgY29uc3QgaW5jbHVkZU9wdGltaXplciA9IGNvbmZpZyA9PSBudWxsID8gZmFsc2UgOiBjb25maWcuaW5jbHVkZU9wdGltaXplcjtcbiAgICBpZiAoaW5jbHVkZU9wdGltaXplciAmJiB0aGlzLm9wdGltaXplciAhPSBudWxsKSB7XG4gICAgICBtb2RlbEFydGlmYWN0cy50cmFpbmluZ0NvbmZpZyA9IHRoaXMuZ2V0VHJhaW5pbmdDb25maWcoKTtcbiAgICAgIGNvbnN0IHdlaWdodFR5cGUgPSAnb3B0aW1pemVyJztcbiAgICAgIGNvbnN0IHtkYXRhOiBvcHRpbWl6ZXJXZWlnaHREYXRhLCBzcGVjczogb3B0aW1pemVyV2VpZ2h0U3BlY3N9ID1cbiAgICAgICAgICBhd2FpdCBpby5lbmNvZGVXZWlnaHRzKGF3YWl0IHRoaXMub3B0aW1pemVyLmdldFdlaWdodHMoKSwgd2VpZ2h0VHlwZSk7XG4gICAgICB3ZWlnaHREYXRhQW5kU3BlY3Muc3BlY3MucHVzaCguLi5vcHRpbWl6ZXJXZWlnaHRTcGVjcyk7XG4gICAgICB3ZWlnaHREYXRhQW5kU3BlY3MuZGF0YSA9IGlvLmNvbmNhdGVuYXRlQXJyYXlCdWZmZXJzKFxuICAgICAgICAgIFt3ZWlnaHREYXRhQW5kU3BlY3MuZGF0YSwgb3B0aW1pemVyV2VpZ2h0RGF0YV0pO1xuICAgIH1cblxuICAgIGlmICh0aGlzLnVzZXJEZWZpbmVkTWV0YWRhdGEgIT0gbnVsbCkge1xuICAgICAgLy8gQ2hlY2sgc2VyaWFsaXplZCBzaXplIG9mIHVzZXItZGVmaW5lZCBtZXRhZGF0YS5cbiAgICAgIGNvbnN0IGNoZWNrU2l6ZSA9IHRydWU7XG4gICAgICBjaGVja1VzZXJEZWZpbmVkTWV0YWRhdGEodGhpcy51c2VyRGVmaW5lZE1ldGFkYXRhLCB0aGlzLm5hbWUsIGNoZWNrU2l6ZSk7XG4gICAgICBtb2RlbEFydGlmYWN0cy51c2VyRGVmaW5lZE1ldGFkYXRhID0gdGhpcy51c2VyRGVmaW5lZE1ldGFkYXRhO1xuICAgIH1cblxuICAgIG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEgPSB3ZWlnaHREYXRhQW5kU3BlY3MuZGF0YTtcbiAgICBtb2RlbEFydGlmYWN0cy53ZWlnaHRTcGVjcyA9IHdlaWdodERhdGFBbmRTcGVjcy5zcGVjcztcbiAgICByZXR1cm4gaGFuZGxlck9yVVJMLnNhdmUobW9kZWxBcnRpZmFjdHMpO1xuICB9XG5cbiAgLyoqXG4gICAqIFNldCB1c2VyLWRlZmluZWQgbWV0YWRhdGEuXG4gICAqXG4gICAqIFRoZSBzZXQgbWV0YWRhdGEgd2lsbCBiZSBzZXJpYWxpemVkIHRvZ2V0aGVyIHdpdGggdGhlIHRvcG9sb2d5XG4gICAqIGFuZCB3ZWlnaHRzIG9mIHRoZSBtb2RlbCBkdXJpbmcgYHNhdmUoKWAgY2FsbHMuXG4gICAqXG4gICAqIEBwYXJhbSBzZXRVc2VyRGVmaW5lZE1ldGFkYXRhXG4gICAqL1xuICBzZXRVc2VyRGVmaW5lZE1ldGFkYXRhKHVzZXJEZWZpbmVkTWV0YWRhdGE6IHt9KTogdm9pZCB7XG4gICAgY2hlY2tVc2VyRGVmaW5lZE1ldGFkYXRhKHVzZXJEZWZpbmVkTWV0YWRhdGEsIHRoaXMubmFtZSk7XG4gICAgdGhpcy51c2VyRGVmaW5lZE1ldGFkYXRhID0gdXNlckRlZmluZWRNZXRhZGF0YTtcbiAgfVxuXG4gIC8qKlxuICAgKiBHZXQgdXNlci1kZWZpbmVkIG1ldGFkYXRhLlxuICAgKlxuICAgKiBUaGUgbWV0YWRhdGEgaXMgc3VwcGxpZWQgdmlhIG9uZSBvZiB0aGUgdHdvIHJvdXRlczpcbiAgICogICAxLiBCeSBjYWxsaW5nIGBzZXRVc2VyRGVmaW5lZE1ldGFkYXRhKClgLlxuICAgKiAgIDIuIExvYWRlZCBkdXJpbmcgbW9kZWwgbG9hZGluZyAoaWYgdGhlIG1vZGVsIGlzIGNvbnN0cnVjdGVkXG4gICAqICAgICAgdmlhIGB0Zi5sb2FkTGF5ZXJzTW9kZWwoKWAuKVxuICAgKlxuICAgKiBJZiBubyB1c2VyLWRlZmluZWQgbWV0YWRhdGEgaXMgYXZhaWxhYmxlIGZyb20gZWl0aGVyIG9mIHRoZVxuICAgKiB0d28gcm91dGVzLCB0aGlzIGZ1bmN0aW9uIHdpbGwgcmV0dXJuIGB1bmRlZmluZWRgLlxuICAgKi9cbiAgZ2V0VXNlckRlZmluZWRNZXRhZGF0YSgpOiB7fSB7XG4gICAgcmV0dXJuIHRoaXMudXNlckRlZmluZWRNZXRhZGF0YTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKExheWVyc01vZGVsKTtcblxuLyoqXG4gKiBBIGB0Zi5GdW5jdGlvbmFsYCBpcyBhbiBhbGlhcyB0byBgdGYuTGF5ZXJzTW9kZWxgLlxuICpcbiAqIFNlZSBhbHNvOlxuICogICBgdGYuTGF5ZXJzTW9kZWxgLCBgdGYuU2VxdWVudGlhbGAsIGB0Zi5sb2FkTGF5ZXJzTW9kZWxgLlxuICovXG4vKiogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ30gKi9cbmV4cG9ydCBjbGFzcyBGdW5jdGlvbmFsIGV4dGVuZHMgTGF5ZXJzTW9kZWwge1xuICBzdGF0aWMgb3ZlcnJpZGUgY2xhc3NOYW1lID0gJ0Z1bmN0aW9uYWwnO1xufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEZ1bmN0aW9uYWwpO1xuIl19