/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* Original source keras/models.py */
import { dispose, io, serialization, util } from '@tensorflow/tfjs-core';
import { getUid } from './backend/state';
import { Input } from './engine/input_layer';
import { getSourceInputs, Node } from './engine/topology';
import { LayersModel } from './engine/training';
import { NotImplementedError, RuntimeError, ValueError } from './errors';
import { deserialize } from './layers/serialization';
import * as generic_utils from './utils/generic_utils';
import { convertPythonicToTs } from './utils/serialization_utils';
import { getExactlyOneShape } from './utils/types_utils';
/**
 * Parses a JSON model configuration file and returns a model instance.
 *
 * ```js
 * // This example shows how to serialize a model using `toJSON()` and
 * // deserialize it as another model using `tf.models.modelFromJSON()`.
 * // Note: this example serializes and deserializes only the topology
 * // of the model; the weights of the loaded model will be different
 * // from those of the the original model, due to random weight
 * // initialization.
 * // To load the topology and weights of a model, use `tf.loadLayersModel()`.
 * const model1 = tf.sequential();
 * model1.add(tf.layers.repeatVector({inputShape: [2], n: 4}));
 * // Serialize `model1` as a JSON object.
 * const model1JSON = model1.toJSON(null, false);
 * model1.summary();
 *
 * const model2 = await tf.models.modelFromJSON(model1JSON);
 * model2.summary();
 * ```
 *
 *  @param modelAndWeightsConfig JSON object or string encoding a model and
 *       weights configuration. It can also be only the topology JSON of the
 *       model, in which case the weights will not be loaded.
 *  @param custom_objects Optional dictionary mapping names
 *       (strings) to custom classes or functions to be
 *       considered during deserialization.
 * @returns A TensorFlow.js Layers `tf.LayersModel` instance (uncompiled).
 */
export async function modelFromJSON(modelAndWeightsConfig, customObjects) {
    if (!('modelTopology' in modelAndWeightsConfig)) {
        modelAndWeightsConfig = { modelTopology: modelAndWeightsConfig };
    }
    modelAndWeightsConfig = modelAndWeightsConfig;
    let modelTopology = modelAndWeightsConfig.modelTopology;
    if (modelTopology['model_config'] != null) {
        // If the model-topology JSON contains a 'model_config' field, then it is
        // a full model JSON (e.g., from `keras.Model.save()`), which contains
        // not only the model's architecture in its 'model_config' field, but
        // additional information such as the model's optimizer. We use only the
        // 'model_config' field currently.
        modelTopology = modelTopology['model_config'];
    }
    const tsConfig = convertPythonicToTs(modelTopology);
    const model = deserialize(tsConfig, customObjects);
    if (modelAndWeightsConfig.weightsManifest != null) {
        // Load the weight values keyed by the original tensor names in the model
        // file that was loaded.  These should match the keys of the weight
        // manifest.
        const weightValues = await io.loadWeights(modelAndWeightsConfig.weightsManifest, modelAndWeightsConfig.pathPrefix, model.weights.map(weight => weight.originalName));
        // Map the weights to the unique tensor names generated during model loading
        const uniqueWeightValues = {};
        for (const weight of model.weights) {
            uniqueWeightValues[weight.originalName] =
                weightValues[weight.originalName];
        }
        model.loadWeights(uniqueWeightValues);
        // Dispose temporary weight values.
        dispose(weightValues);
    }
    return model;
}
/**
 * Load a model composed of Layer objects, including its topology and optionally
 * weights. See the Tutorial named "How to import a Keras Model" for usage
 * examples.
 *
 * This method is applicable to:
 *
 * 1. Models created with the `tf.layers.*`, `tf.sequential`, and
 * `tf.model` APIs of TensorFlow.js and later saved with the
 * `tf.LayersModel.save` method.
 * 2. Models converted from Keras or TensorFlow tf.keras using the
 * [tensorflowjs_converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter).
 *
 * This mode is *not* applicable to TensorFlow `SavedModel`s or their converted
 * forms. For those models, use `tf.loadGraphModel`.
 *
 * Example 1. Load a model from an HTTP server.
 *
 * ```js
 * const model = await tf.loadLayersModel(
 *     'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json');
 * model.summary();
 * ```
 *
 * Example 2: Save `model`'s topology and weights to browser [local
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
 * Example 3. Saving `model`'s topology and weights to browser
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
 * Example 4. Load a model from user-selected files from HTML
 * [file input
 * elements](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/file).
 *
 * ```js
 * // Note: this code snippet will not work without the HTML elements in the
 * //   page
 * const jsonUpload = document.getElementById('json-upload');
 * const weightsUpload = document.getElementById('weights-upload');
 *
 * const model = await tf.loadLayersModel(
 *     tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));
 * ```
 *
 * @param pathOrIOHandler Can be either of the two formats
 *   1. A string path to the `ModelAndWeightsConfig` JSON describing
 *      the model in the canonical TensorFlow.js format. For file://
 *      (tfjs-node-only), http:// and https:// schemas, the path can be
 *      either absolute or relative. The content of the JSON file is assumed to
 *      be a JSON object with the following fields and values:
 *      - 'modelTopology': A JSON object that can be either of:
 *        1. a model architecture JSON consistent with the format of the return
 *            value of `keras.Model.to_json()`
 *        2. a full model JSON in the format of `keras.models.save_model()`.
 *      - 'weightsManifest': A TensorFlow.js weights manifest.
 *      See the Python converter function `save_model()` for more details.
 *      It is also assumed that model weights can be accessed from relative
 *      paths described by the `paths` fields in weights manifest.
 *   2. A `tf.io.IOHandler` object that loads model artifacts with its `load`
 *      method.
 * @param options Optional configuration arguments for the model loading,
 *   including:
 *   - `strict`: Require that the provided weights exactly match those required
 *     by the layers.  Default true.  Passing false means that both extra
 *     weights and missing weights will be silently ignored.
 *   - `onProgress`: A progress callback of the form:
 *     `(fraction: number) => void`. This callback can be used to monitor the
 *     model-loading process.
 * @returns A `Promise` of `tf.LayersModel`, with the topology and weights
 *     loaded.
 *
 * @doc {heading: 'Models', subheading: 'Loading'}
 */
export async function loadLayersModel(pathOrIOHandler, options) {
    if (options == null) {
        options = {};
    }
    if (typeof pathOrIOHandler === 'string') {
        const handlers = io.getLoadHandlers(pathOrIOHandler, options);
        if (handlers.length === 0) {
            // For backward compatibility: if no load handler can be found,
            // assume it is a relative http path.
            // TODO(cais): Reformat the args into a single `LoadOptions` once the core
            // is refactored.
            handlers.push(io.browserHTTPRequest(pathOrIOHandler, options));
        }
        else if (handlers.length > 1) {
            throw new ValueError(`Found more than one (${handlers.length}) load handlers for ` +
                `URL '${pathOrIOHandler}'`);
        }
        pathOrIOHandler = handlers[0];
    }
    return loadLayersModelFromIOHandler(pathOrIOHandler, undefined, options);
}
/**
 * Load a model and optionally its weights, using an IOHandler object.
 *
 * @param handler The instance of `IOHandler` to be used during the model
 *   loading.
 * @param customObjects Any optional custom objects to be used during model
 *   loading.
 * @param strict Whether the weight loading will be done in strict mode.
 *   Default: `true`.
 */
export async function loadLayersModelFromIOHandler(handler, customObjects, options) {
    if (options == null) {
        options = {};
    }
    if (handler.load == null) {
        throw new ValueError('Cannot proceed with model loading because the IOHandler provided ' +
            'does not have the `load` method implemented.');
    }
    const artifacts = await handler.load();
    let modelTopology = artifacts.modelTopology;
    if (modelTopology['model_config'] != null) {
        modelTopology = modelTopology['model_config'];
    }
    const strict = options.strict == null ? true : options.strict;
    // If weights are provided and the weight-loading mode is strict, use
    // fast weight initialization. This skips costly initializers such as
    // 'orthogonal' and saves unnecessary computation in cases where
    // the initialized weight values will immediately be overwritten by
    // loaded weight values.
    const fastWeightInit = artifacts.weightData != null && artifacts.weightSpecs != null && strict;
    const model = deserialize(convertPythonicToTs(modelTopology), customObjects, fastWeightInit);
    const trainingConfig = artifacts.trainingConfig;
    if (trainingConfig != null) {
        model.loadTrainingConfig(trainingConfig);
    }
    if (artifacts.userDefinedMetadata != null) {
        model.setUserDefinedMetadata(artifacts.userDefinedMetadata);
    }
    // If weightData is present, load the weights into the model.
    if (artifacts.weightData != null) {
        // Loading weights requires weightSpecs.
        if (artifacts.weightSpecs == null) {
            throw new ValueError('LayersModel artifacts contains weight data, but not weight specs. ' +
                'Therefore loading of weights cannot proceed.');
        }
        const { modelWeights, optimizerWeights } = decodeModelAndOptimizerWeights(artifacts.weightData, artifacts.weightSpecs);
        model.loadWeights(modelWeights, strict);
        if (model.optimizer != null && optimizerWeights.length > 0) {
            await model.optimizer.setWeights(optimizerWeights);
        }
        // Dispose temporary weight values.
        dispose(modelWeights);
        dispose(optimizerWeights.map(w => w.tensor));
    }
    return model;
}
function decodeModelAndOptimizerWeights(buffer, specs) {
    const name2Tensor = io.decodeWeights(buffer, specs);
    const modelWeights = {};
    const optimizerWeights = [];
    specs.forEach(spec => {
        if (spec.group === 'optimizer') {
            optimizerWeights.push({ name: spec.name, tensor: name2Tensor[spec.name] });
        }
        else {
            modelWeights[spec.name] = name2Tensor[spec.name];
        }
    });
    return { modelWeights, optimizerWeights };
}
/**
 * A model with a stack of layers, feeding linearly from one to the next.
 *
 * `tf.sequential` is a factory function that creates an instance of
 * `tf.Sequential`.
 *
 * ```js
 *  // Define a model for linear regression.
 *  const model = tf.sequential();
 *  model.add(tf.layers.dense({units: 1, inputShape: [1]}));
 *
 *  // Prepare the model for training: Specify the loss and the optimizer.
 *  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 *
 *  // Generate some synthetic data for training.
 *  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
 *  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
 *
 *  // Train the model using the data then do inference on a data point the
 *  // model hasn't seen:
 *  await model.fit(xs, ys);
 *  model.predict(tf.tensor2d([5], [1, 1])).print();
 * ```
 *
 * @doc {heading: 'Models', subheading: 'Classes'}
 */
export class Sequential extends LayersModel {
    constructor(args) {
        super({ inputs: [], outputs: [] });
        args = args || {};
        this.trainable = true;
        this.built = false;
        // Set model name.
        this.name = (args.name != null) ? args.name : getUid('sequential_');
        // Add to the model any layers passed to the constructor.
        if (args.layers != null) {
            for (const layer of args.layers) {
                this.add(layer);
            }
        }
    }
    // Helper function to Sequential.add  Throws if the new output shape will be
    // invalid.
    checkShape(layer) {
        const shape = layer.inboundNodes[0].outputTensors[0].shape;
        if (shape.some(x => x < 0)) {
            throw new ValueError('Negative dimension size caused by adding layer ' +
                `${layer.name} with input shape [` +
                `${layer.inboundNodes[0].inputTensors[0].shape}]`);
        }
    }
    /**
     * Adds a layer instance on top of the layer stack.
     *
     * ```js
     *  const model = tf.sequential();
     *  model.add(tf.layers.dense({units: 8, inputShape: [1]}));
     *  model.add(tf.layers.dense({units: 4, activation: 'relu6'}));
     *  model.add(tf.layers.dense({units: 1, activation: 'relu6'}));
     *  // Note that the untrained model is random at this point.
     *  model.predict(tf.randomNormal([10, 1])).print();
     * ```
     * @param layer Layer instance.
     *
     * @exception ValueError In case the `layer` argument does not know its
     * input shape.
     * @exception ValueError In case the `layer` argument has multiple output
     *   tensors, or is already connected somewhere else (forbidden in
     *   `Sequential` models).
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    add(layer) {
        const isLayerModelInstance = layer instanceof Sequential || layer instanceof LayersModel;
        let modelLayer;
        if (isLayerModelInstance) {
            modelLayer = layer;
            if (modelLayer.outputs.length !== 1) {
                throw new ValueError('All layers in a Sequential model ' +
                    'should have a single output tensor. ' +
                    'For multi-output layers, ' +
                    'use the functional API.');
            }
            if (modelLayer.inputs.length !== 1) {
                throw new ValueError('All layers in a Sequential model ' +
                    'should have a single input tensor. ' +
                    'For multi-input layers, ' +
                    'use the functional API.');
            }
        }
        if (this.outputs.length === 0) {
            // first layer in model: check that it is an input layer
            if (layer.inboundNodes.length === 0) {
                // create an input layer
                if (layer.batchInputShape == null) {
                    throw new ValueError('The first layer in a Sequential model must ' +
                        'get an `inputShape` or `batchInputShape` argument.');
                }
                // Instantiate the input layer.
                const x = Input({
                    batchShape: layer.batchInputShape,
                    dtype: layer.dtype,
                    name: layer.name + '_input'
                });
                // This will build the current layer and create the node connecting
                // the current layer to the input layer we just created.
                layer.apply(x);
            }
            if (isLayerModelInstance) {
                this.outputs = modelLayer.outputs;
                this.inputs = modelLayer.inputs;
            }
            else {
                if (layer.inboundNodes.length !== 1) {
                    throw new ValueError('A layer added to a Sequential model must not already be ' +
                        `connected somewhere else. LayersModel received layer ${layer.name} ` +
                        `which has ${layer.inboundNodes.length} pre-existing inbound ` +
                        'connections.');
                }
                if (layer.inboundNodes[0].outputTensors.length !== 1) {
                    throw new ValueError('All layers in a Sequential model ' +
                        'should have a single output tensor. ' +
                        'For multi-output layers, ' +
                        'use the functional API.');
                }
                this.checkShape(layer);
                this.outputs = [layer.inboundNodes[0].outputTensors[0]];
                this.inputs = getSourceInputs(this.outputs[0]);
            }
            this.inboundNodes = [];
            // We create an input node, which we will keep updated
            // as we add more layers.
            // (This call has side effects.)
            // tslint:disable-next-line:no-unused-expression
            new Node({
                outboundLayer: this,
                inboundLayers: [],
                nodeIndices: [],
                tensorIndices: [],
                inputTensors: this.inputs,
                outputTensors: this.outputs,
                // no model-level masking for now
                inputMasks: generic_utils.pyListRepeat(null, this.inputs.length),
                outputMasks: [null],
                inputShapes: this.inputs.map(x => x.shape),
                outputShapes: this.outputs[0].shape
            });
        }
        else {
            const outputTensor = layer.apply(this.outputs[0]);
            if (Array.isArray(outputTensor)) {
                throw new TypeError('All layers in a Sequential model ' +
                    'should have a single output tensor. ' +
                    'For multi-output layers, ' +
                    'use the functional API.');
            }
            this.checkShape(layer);
            this.outputs = [outputTensor];
            // update self.inbound_nodes
            this.inboundNodes[0].outputTensors = this.outputs;
            this.inboundNodes[0].outputShapes = [this.outputs[0].shape];
        }
        this.layers.push(layer);
        this.built = false;
    }
    /**
     * Removes the last layer in the model.
     *
     * @exception TypeError if there are no layers in the model.
     */
    pop() {
        if (this.layers.length === 0) {
            throw new TypeError('There are no layers in the model.');
        }
        this.layers.pop();
        if (this.layers.length === 0) {
            this.outputs = [];
            this.inboundNodes = [];
            this.outboundNodes = [];
        }
        else {
            const lastLayerIndex = this.layers.length - 1;
            this.layers[lastLayerIndex].outboundNodes = [];
            this.outputs = [this.layers[lastLayerIndex].output];
            // update self.inbound_nodes
            this.inboundNodes[0].outputTensors = this.outputs;
            this.inboundNodes[0].outputShapes = [this.outputs[0].shape];
        }
    }
    call(inputs, kwargs) {
        if (this.model == null) {
            this.build();
        }
        return this.model.call(inputs, kwargs);
    }
    build(inputShape) {
        // Call `getExactlyOneShape` without using its return value,
        // to verify that exactly one input shape is provided.
        getExactlyOneShape(inputShape);
        if (this.inputs.length === 0 || this.outputs.length === 0) {
            throw new TypeError('Sequential model cannot be built: model is empty.' +
                ' Add some layers first.');
        }
        // actually create the model
        this.model = new LayersModel({
            inputs: this.inputs,
            outputs: this.outputs[0],
            name: this.name + '_model'
        });
        this.model.trainable = this.trainable;
        // mirror model attributes
        this.supportsMasking = this.model.supportsMasking;
        // TODO(michaelterry): Add caches
        this.inputLayers = this.model.inputLayers;
        this.inputLayersNodeIndices = this.model.inputLayersNodeIndices;
        this.inputLayersTensorIndices = this.model.inputLayersTensorIndices;
        this.outputLayers = this.model.outputLayers;
        this.outputLayersNodeIndices = this.model.outputLayersNodeIndices;
        this.outputLayersTensorIndices = this.model.outputLayersTensorIndices;
        this.nodesByDepth = this.model.nodesByDepth;
        this.containerNodes = this.model.containerNodes;
        this.outputNames = this.model.outputNames;
        this.inputNames = this.model.inputNames;
        // TODO(michaelterry): Add feedInputNames, feedInputs, if needed.
        // TODO(michaelterry): Add callbackModel if needed.
        this.built = true;
    }
    countParams() {
        if (!this.built) {
            this.build();
        }
        return super.countParams();
    }
    /**
     * Print a text summary of the Sequential model's layers.
     *
     * The summary includes
     * - Name and type of all layers that comprise the model.
     * - Output shape(s) of the layers
     * - Number of weight parameters of each layer
     * - The total number of trainable and non-trainable parameters of the
     * model.
     *
     * ```js
     * const model = tf.sequential();
     * model.add(
     *     tf.layers.dense({units: 100, inputShape: [10], activation: 'relu'}));
     * model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
     *
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
            this.build();
        }
        super.summary(lineLength, positions, printFn);
    }
    /**
     * Sets the weights of the model.
     *
     * @param weights Should be a list of Tensors with shapes and types matching
     *   the output of `model.getWeights()`.
     */
    setWeights(weights) {
        if (this.model == null) {
            this.build();
        }
        this.model.setWeights(weights);
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
     * const result = model.evaluate(tf.ones([8, 10]), tf.ones([8, 1]), {
     *   batchSize: 4,
     * });
     * result.print();
     * ```
     *
     * @param x `tf.Tensor` of test data, or an `Array` of `tf.Tensor`s if the
     * model has multiple inputs.
     * @param y `tf.Tensor` of target data, or an `Array` of `tf.Tensor`s if the
     * model has multiple outputs.
     * @param args A `ModelEvaluateConfig`, containing optional fields.
     *
     * @return `Scalar` test loss (if the model has a single output and no
     *   metrics) or `Array` of `Scalar`s (if the model has multiple outputs
     *   and/or metrics). The attribute `model.metricsNames`
     *   will give you the display labels for the scalar outputs.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    evaluate(x, y, args = {}) {
        if (!this.built) {
            throw new RuntimeError('The model needs to be compiled before being used.');
        }
        return this.model.evaluate(x, y, args);
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
        if (!this.built) {
            throw new RuntimeError('The model needs to be compiled before being used.');
        }
        return this.model.evaluateDataset(dataset, args);
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
     * model.predict(tf.ones([2, 10])).print();
     * ```
     *
     * @param x The input data, as a Tensor, or an `Array` of `tf.Tensor`s if
     *   the model has multiple inputs.
     * @param conifg A `ModelPredictConfig` object containing optional fields.
     *
     * @return `tf.Tensor`(s) of predictions.
     *
     * @exception ValueError In case of mismatch between the provided input data
     *   and the model's expectations, or in case a stateful model receives a
     *   number of samples that is not a multiple of the batch size.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    predict(x, args = {}) {
        if (this.model == null) {
            this.build();
        }
        return this.model.predict(x, args);
    }
    /**
     * Returns predictions for a single batch of samples.
     *
     * @param x: Input samples, as a Tensor, or list of Tensors (if the model
     *   has multiple inputs).
     * @return Tensor(s) of predictions
     */
    predictOnBatch(x) {
        if (this.model == null) {
            this.build();
        }
        return this.model.predictOnBatch(x);
    }
    /**
     * See `LayersModel.compile`.
     *
     * @param args
     */
    compile(args) {
        this.build();
        this.model.compile(args);
        this.optimizer_ = this.model.optimizer;
        // tslint:disable-next-line:no-any
        this.isOptimizerOwned = this.model.isOptimizerOwned;
        this.loss = this.model.loss;
        this.metrics = this.model.metrics;
        // TODO(cais): Add this.lossWeights, this.sampleWeightMode,
        //   this.weightedMetrics, this.targets.
        this.metricsTensors = this.model.metricsTensors;
        this.metricsNames = this.model.metricsNames;
        // TODO(cais): Add sampleWeights.
    }
    get optimizer() {
        return this.model == null ? undefined : this.model.optimizer;
    }
    set optimizer(optimizer) {
        this.model.optimizer = optimizer;
    }
    /**
     * Trains the model for a fixed number of epochs (iterations on a dataset).
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * const history = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
     *   batchSize: 4,
     *   epochs: 3
     * });
     * console.log(history.history.loss[0]);
     * ```
     *
     * @param x `tf.Tensor` of training data, or an array of `tf.Tensor`s if the
     * model has multiple inputs. If all inputs in the model are named, you can
     * also pass a dictionary mapping input names to `tf.Tensor`s.
     * @param y `tf.Tensor` of target (label) data, or an array of `tf.Tensor`s if
     * the model has multiple outputs. If all outputs in the model are named, you
     *  can also pass a dictionary mapping output names to `tf.Tensor`s.
     * @param args  A `ModelFitConfig`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @exception ValueError In case of mismatch between the provided input data
     *   and what the model expects.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    async fit(x, y, args = {}) {
        if (!this.built) {
            throw new RuntimeError('The model needs to be compiled before ' +
                'being used.');
        }
        return this.model.fit(x, y, args);
    }
    /**
     * Trains the model using a dataset object.
     *
     * ```js
     * const xArray = [
     *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
     *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
     *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
     *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
     * ];
     * const yArray = [1, 1, 1, 1];
     * // Create a dataset from the JavaScript array.
     * const xDataset = tf.data.array(xArray);
     * const yDataset = tf.data.array(yArray);
     * // Zip combines the `x` and `y` Datasets into a single Dataset, the
     * // iterator of which will return an object containing of two tensors,
     * // corresponding to `x` and `y`.  The call to `batch(4)` will bundle
     * // four such samples into a single object, with the same keys now pointing
     * // to tensors that hold 4 examples, organized along the batch dimension.
     * // The call to `shuffle(4)` causes each iteration through the dataset to
     * // happen in a different order.  The size of the shuffle window is 4.
     * const xyDataset = tf.data.zip({xs: xDataset, ys: yDataset})
     *     .batch(4)
     *     .shuffle(4);
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [9]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * const history = await model.fitDataset(xyDataset, {
     *   epochs: 4,
     *   callbacks: {onEpochEnd: (epoch, logs) => console.log(logs.loss)}
     * });
     * ```
     *
     * @param dataset A dataset object. Its `iterator()` method is expected to
     *   generate a dataset iterator object, the `next()` method of which is
     *   expected to produce data batches for evaluation. The return value of the
     *   `next()` call ought to contain a boolean `done` field and a `value`
     *   field.
     *
     *   The `value` field is expected to be an object of with fields
     *   `xs` and `ys`, which point to the feature tensor and the target tensor,
     *   respectively. This case is for models with exactly one input and one
     *   output (e.g. a sequential model). For example:
     *   ```js
     *   {value: {xs: xsTensor, ys: ysTensor}, done: false}
     *   ```
     *
     *   If the model has multiple inputs, the `xs` field of `value` should
     *   be an object mapping input names to their respective feature tensors.
     *   For example:
     *   ```js
     *   {
     *     value: {
     *       xs: {
     *         input_1: xsTensor1,
     *         input_2: xsTensor2
     *       },
     *       ys: ysTensor
     *     },
     *     done: false
     *   }
     *   ```
     *   If the model has multiple outputs, the `ys` field of `value` should
     *   be an object mapping output names to their respective target tensors.
     *   For example:
     *   ```js
     *   {
     *     value: {
     *       xs: xsTensor,
     *       ys: {
     *         output_1: ysTensor1,
     *         output_2: ysTensor2
     *       },
     *     },
     *     done: false
     *   }
     *   ```
     * @param args A `ModelFitDatasetArgs`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @doc {heading: 'Models', subheading: 'Classes', ignoreCI: true}
     */
    async fitDataset(dataset, args) {
        if (!this.built) {
            throw new RuntimeError('The model needs to be compiled before ' +
                'being used.');
        }
        return this.model.fitDataset(dataset, args);
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
        return this.model.trainOnBatch(x, y);
    }
    /* See parent class for JsDoc */
    /** @nocollapse */
    static fromConfig(cls, config, customObjects = {}, fastWeightInit = false) {
        let configArray;
        let extraModelConfig = {};
        if (config instanceof Array) {
            if (!(config[0].className != null) ||
                config[0]['className'] === 'Merge') {
                throw new ValueError('Legacy serialization format not supported yet.');
            }
            configArray = config;
        }
        else {
            util.assert(config['layers'] != null, () => `When the config data for a Sequential model is not an Array, ` +
                `it must be an Object that contains the 'layers' field.`);
            configArray = config['layers'];
            delete config['layers'];
            extraModelConfig = config;
        }
        const model = new cls(extraModelConfig);
        if (!(model instanceof Sequential)) {
            throw new NotImplementedError(`Sequential.fromConfig called on non-Sequential input: ${model}`);
        }
        for (const conf of configArray) {
            const customObjects = undefined;
            const layer = deserialize(conf, customObjects, fastWeightInit);
            if (fastWeightInit) {
                layer.setFastWeightInitDuringBuild(true);
            }
            model.add(layer);
        }
        return model;
    }
    /**
     * Setter used for force stopping of LayersModel.fit() (i.e., training).
     *
     * Example:
     *
     * ```js
     * const model = tf.sequential();
     * model.add(tf.layers.dense({units: 1, inputShape: [10]}));
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
     * // There should be only 3 values in the loss array, instead of 10 values,
     * // due to the stopping after 3 epochs.
     * console.log(history.history.loss);
     * ```
     */
    set stopTraining(stop) {
        // TODO(cais): When refactoring to remove the composition pattern happens,
        // remove this method overriding.
        if (this.model == null) {
            throw new ValueError('Cannot set the stopTraining property of a sequential model before ' +
                'it is compiled.');
        }
        this.model.stopTraining = stop;
    }
    get stopTraining() {
        if (this.model == null) {
            throw new ValueError('Cannot get the stopTraining property of a sequential model before ' +
                'it is compiled.');
        }
        return this.model.stopTraining;
    }
    // TODO(cais): Override get trainableWeights() here
    // tslint:disable-next-line:no-any
    getConfig() {
        // NOTE(cais): We override the return type of getConfig() to `any` here,
        //   because the `Sequential` class is a special case among `Container`
        //   subtypes in that its getConfig() method returns an Array (not a
        //   dict).
        const layers = [];
        for (const layer of this.layers) {
            const dict = {};
            dict['className'] = layer.getClassName();
            dict['config'] = layer.getConfig();
            layers.push(dict);
        }
        return { name: this.name, layers };
    }
}
/** @nocollapse */
Sequential.className = 'Sequential';
serialization.registerClass(Sequential);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibW9kZWxzLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1sYXllcnMvc3JjL21vZGVscy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVILHFDQUFxQztBQUVyQyxPQUFPLEVBQUMsT0FBTyxFQUFFLEVBQUUsRUFBcUMsYUFBYSxFQUFVLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRWxILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUd2QyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sc0JBQXNCLENBQUM7QUFDM0MsT0FBTyxFQUFDLGVBQWUsRUFBUyxJQUFJLEVBQWlCLE1BQU0sbUJBQW1CLENBQUM7QUFDL0UsT0FBTyxFQUFDLFdBQVcsRUFBc0MsTUFBTSxtQkFBbUIsQ0FBQztBQUduRixPQUFPLEVBQUMsbUJBQW1CLEVBQUUsWUFBWSxFQUFFLFVBQVUsRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUl2RSxPQUFPLEVBQUMsV0FBVyxFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFFbkQsT0FBTyxLQUFLLGFBQWEsTUFBTSx1QkFBdUIsQ0FBQztBQUN2RCxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSw2QkFBNkIsQ0FBQztBQUNoRSxPQUFPLEVBQUMsa0JBQWtCLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUV2RDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTRCRztBQUNILE1BQU0sQ0FBQyxLQUFLLFVBQVUsYUFBYSxDQUMvQixxQkFBdUQsRUFDdkQsYUFBd0M7SUFDMUMsSUFBSSxDQUFDLENBQUMsZUFBZSxJQUFJLHFCQUFxQixDQUFDLEVBQUU7UUFDL0MscUJBQXFCLEdBQUcsRUFBQyxhQUFhLEVBQUUscUJBQXFCLEVBQUMsQ0FBQztLQUNoRTtJQUNELHFCQUFxQixHQUFHLHFCQUE4QyxDQUFDO0lBRXZFLElBQUksYUFBYSxHQUFHLHFCQUFx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SSxDQUFDLElBQUksQ0FBQyxDQUFDO1NBQ25CO1FBQ0QsT0FBTyxFQUFDLElBQUksRUFBRSxJQUFJLENBQUMsSUFBSSxFQUFFLE1BQU0sRUFBQyxDQUFDO0lBQ25DLENBQUM7O0FBMXNCRCxrQkFBa0I7QUFDRixvQkFBUyxHQUFHLFlBQVksQ0FBQztBQTJzQjNDLGFBQWEsQ0FBQyxhQUFhLENBQUMsVUFBVSxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlXG4gKiBsaWNlbnNlIHRoYXQgY2FuIGJlIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgb3IgYXRcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vKiBPcmlnaW5hbCBzb3VyY2Uga2VyYXMvbW9kZWxzLnB5ICovXG5cbmltcG9ydCB7ZGlzcG9zZSwgaW8sIE5hbWVkVGVuc29yTWFwLCBPcHRpbWl6ZXIsIFNjYWxhciwgc2VyaWFsaXphdGlvbiwgVGVuc29yLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2dldFVpZH0gZnJvbSAnLi9iYWNrZW5kL3N0YXRlJztcbmltcG9ydCB7SGlzdG9yeX0gZnJvbSAnLi9iYXNlX2NhbGxiYWNrcyc7XG5pbXBvcnQge0RhdGFzZXR9IGZyb20gJy4vZW5naW5lL2RhdGFzZXRfc3R1Yic7XG5pbXBvcnQge0lucHV0fSBmcm9tICcuL2VuZ2luZS9pbnB1dF9sYXllcic7XG5pbXBvcnQge2dldFNvdXJjZUlucHV0cywgTGF5ZXIsIE5vZGUsIFN5bWJvbGljVGVuc29yfSBmcm9tICcuL2VuZ2luZS90b3BvbG9neSc7XG5pbXBvcnQge0xheWVyc01vZGVsLCBNb2RlbENvbXBpbGVBcmdzLCBNb2RlbEV2YWx1YXRlQXJnc30gZnJvbSAnLi9lbmdpbmUvdHJhaW5pbmcnO1xuaW1wb3J0IHtNb2RlbEV2YWx1YXRlRGF0YXNldEFyZ3MsIE1vZGVsRml0RGF0YXNldEFyZ3N9IGZyb20gJy4vZW5naW5lL3RyYWluaW5nX2RhdGFzZXQnO1xuaW1wb3J0IHtNb2RlbEZpdEFyZ3N9IGZyb20gJy4vZW5naW5lL3RyYWluaW5nX3RlbnNvcnMnO1xuaW1wb3J0IHtOb3RJbXBsZW1lbnRlZEVycm9yLCBSdW50aW1lRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4vZXJyb3JzJztcbmltcG9ydCB7U2hhcGV9IGZyb20gJy4va2VyYXNfZm9ybWF0L2NvbW1vbic7XG5pbXBvcnQge1RyYWluaW5nQ29uZmlnfSBmcm9tICcuL2tlcmFzX2Zvcm1hdC90cmFpbmluZ19jb25maWcnO1xuaW1wb3J0IHtQeUpzb25EaWN0fSBmcm9tICcuL2tlcmFzX2Zvcm1hdC90eXBlcyc7XG5pbXBvcnQge2Rlc2VyaWFsaXplfSBmcm9tICcuL2xheWVycy9zZXJpYWxpemF0aW9uJztcbmltcG9ydCB7S3dhcmdzLCBOYW1lZFRlbnNvcn0gZnJvbSAnLi90eXBlcyc7XG5pbXBvcnQgKiBhcyBnZW5lcmljX3V0aWxzIGZyb20gJy4vdXRpbHMvZ2VuZXJpY191dGlscyc7XG5pbXBvcnQge2NvbnZlcnRQeXRob25pY1RvVHN9IGZyb20gJy4vdXRpbHMvc2VyaWFsaXphdGlvbl91dGlscyc7XG5pbXBvcnQge2dldEV4YWN0bHlPbmVTaGFwZX0gZnJvbSAnLi91dGlscy90eXBlc191dGlscyc7XG5cbi8qKlxuICogUGFyc2VzIGEgSlNPTiBtb2RlbCBjb25maWd1cmF0aW9uIGZpbGUgYW5kIHJldHVybnMgYSBtb2RlbCBpbnN0YW5jZS5cbiAqXG4gKiBgYGBqc1xuICogLy8gVGhpcyBleGFtcGxlIHNob3dzIGhvdyB0byBzZXJpYWxpemUgYSBtb2RlbCB1c2luZyBgdG9KU09OKClgIGFuZFxuICogLy8gZGVzZXJpYWxpemUgaXQgYXMgYW5vdGhlciBtb2RlbCB1c2luZyBgdGYubW9kZWxzLm1vZGVsRnJvbUpTT04oKWAuXG4gKiAvLyBOb3RlOiB0aGlzIGV4YW1wbGUgc2VyaWFsaXplcyBhbmQgZGVzZXJpYWxpemVzIG9ubHkgdGhlIHRvcG9sb2d5XG4gKiAvLyBvZiB0aGUgbW9kZWw7IHRoZSB3ZWlnaHRzIG9mIHRoZSBsb2FkZWQgbW9kZWwgd2lsbCBiZSBkaWZmZXJlbnRcbiAqIC8vIGZyb20gdGhvc2Ugb2YgdGhlIHRoZSBvcmlnaW5hbCBtb2RlbCwgZHVlIHRvIHJhbmRvbSB3ZWlnaHRcbiAqIC8vIGluaXRpYWxpemF0aW9uLlxuICogLy8gVG8gbG9hZCB0aGUgdG9wb2xvZ3kgYW5kIHdlaWdodHMgb2YgYSBtb2RlbCwgdXNlIGB0Zi5sb2FkTGF5ZXJzTW9kZWwoKWAuXG4gKiBjb25zdCBtb2RlbDEgPSB0Zi5zZXF1ZW50aWFsKCk7XG4gKiBtb2RlbDEuYWRkKHRmLmxheWVycy5yZXBlYXRWZWN0b3Ioe2lucHV0U2hhcGU6IFsyXSwgbjogNH0pKTtcbiAqIC8vIFNlcmlhbGl6ZSBgbW9kZWwxYCBhcyBhIEpTT04gb2JqZWN0LlxuICogY29uc3QgbW9kZWwxSlNPTiA9IG1vZGVsMS50b0pTT04obnVsbCwgZmFsc2UpO1xuICogbW9kZWwxLnN1bW1hcnkoKTtcbiAqXG4gKiBjb25zdCBtb2RlbDIgPSBhd2FpdCB0Zi5tb2RlbHMubW9kZWxGcm9tSlNPTihtb2RlbDFKU09OKTtcbiAqIG1vZGVsMi5zdW1tYXJ5KCk7XG4gKiBgYGBcbiAqXG4gKiAgQHBhcmFtIG1vZGVsQW5kV2VpZ2h0c0NvbmZpZyBKU09OIG9iamVjdCBvciBzdHJpbmcgZW5jb2RpbmcgYSBtb2RlbCBhbmRcbiAqICAgICAgIHdlaWdodHMgY29uZmlndXJhdGlvbi4gSXQgY2FuIGFsc28gYmUgb25seSB0aGUgdG9wb2xvZ3kgSlNPTiBvZiB0aGVcbiAqICAgICAgIG1vZGVsLCBpbiB3aGljaCBjYXNlIHRoZSB3ZWlnaHRzIHdpbGwgbm90IGJlIGxvYWRlZC5cbiAqICBAcGFyYW0gY3VzdG9tX29iamVjdHMgT3B0aW9uYWwgZGljdGlvbmFyeSBtYXBwaW5nIG5hbWVzXG4gKiAgICAgICAoc3RyaW5ncykgdG8gY3VzdG9tIGNsYXNzZXMgb3IgZnVuY3Rpb25zIHRvIGJlXG4gKiAgICAgICBjb25zaWRlcmVkIGR1cmluZyBkZXNlcmlhbGl6YXRpb24uXG4gKiBAcmV0dXJucyBBIFRlbnNvckZsb3cuanMgTGF5ZXJzIGB0Zi5MYXllcnNNb2RlbGAgaW5zdGFuY2UgKHVuY29tcGlsZWQpLlxuICovXG5leHBvcnQgYXN5bmMgZnVuY3Rpb24gbW9kZWxGcm9tSlNPTihcbiAgICBtb2RlbEFuZFdlaWdodHNDb25maWc6IE1vZGVsQW5kV2VpZ2h0c0NvbmZpZ3xQeUpzb25EaWN0LFxuICAgIGN1c3RvbU9iamVjdHM/OiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QpOiBQcm9taXNlPExheWVyc01vZGVsPiB7XG4gIGlmICghKCdtb2RlbFRvcG9sb2d5JyBpbiBtb2RlbEFuZFdlaWdodHNDb25maWcpKSB7XG4gICAgbW9kZWxBbmRXZWlnaHRzQ29uZmlnID0ge21vZGVsVG9wb2xvZ3k6IG1vZGVsQW5kV2VpZ2h0c0NvbmZpZ307XG4gIH1cbiAgbW9kZWxBbmRXZWlnaHRzQ29uZmlnID0gbW9kZWxBbmRXZWlnaHRzQ29uZmlnIGFzIE1vZGVsQW5kV2VpZ2h0c0NvbmZpZztcblxuICBsZXQgbW9kZWxUb3BvbG9neSA9IG1vZGVsQW5kV2VpZ2h0c0NvbmZpZy5tb2RlbFRvcG9sb2d5O1xuICBpZiAobW9kZWxUb3BvbG9neVsnbW9kZWxfY29uZmlnJ10gIT0gbnVsbCkge1xuICAgIC8vIElmIHRoZSBtb2RlbC10b3BvbG9neSBKU09OIGNvbnRhaW5zIGEgJ21vZGVsX2NvbmZpZycgZmllbGQsIHRoZW4gaXQgaXNcbiAgICAvLyBhIGZ1bGwgbW9kZWwgSlNPTiAoZS5nLiwgZnJvbSBga2VyYXMuTW9kZWwuc2F2ZSgpYCksIHdoaWNoIGNvbnRhaW5zXG4gICAgLy8gbm90IG9ubHkgdGhlIG1vZGVsJ3MgYXJjaGl0ZWN0dXJlIGluIGl0cyAnbW9kZWxfY29uZmlnJyBmaWVsZCwgYnV0XG4gICAgLy8gYWRkaXRpb25hbCBpbmZvcm1hdGlvbiBzdWNoIGFzIHRoZSBtb2RlbCdzIG9wdGltaXplci4gV2UgdXNlIG9ubHkgdGhlXG4gICAgLy8gJ21vZGVsX2NvbmZpZycgZmllbGQgY3VycmVudGx5LlxuICAgIG1vZGVsVG9wb2xvZ3kgPSBtb2RlbFRvcG9sb2d5Wydtb2RlbF9jb25maWcnXSBhcyBQeUpzb25EaWN0O1xuICB9XG4gIGNvbnN0IHRzQ29uZmlnID1cbiAgICAgIGNvbnZlcnRQeXRob25pY1RvVHMobW9kZWxUb3BvbG9neSkgYXMgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0O1xuICBjb25zdCBtb2RlbCA9IGRlc2VyaWFsaXplKHRzQ29uZmlnLCBjdXN0b21PYmplY3RzKSBhcyBMYXllcnNNb2RlbDtcblxuICBpZiAobW9kZWxBbmRXZWlnaHRzQ29uZmlnLndlaWdodHNNYW5pZmVzdCAhPSBudWxsKSB7XG4gICAgLy8gTG9hZCB0aGUgd2VpZ2h0IHZhbHVlcyBrZXllZCBieSB0aGUgb3JpZ2luYWwgdGVuc29yIG5hbWVzIGluIHRoZSBtb2RlbFxuICAgIC8vIGZpbGUgdGhhdCB3YXMgbG9hZGVkLiAgVGhlc2Ugc2hvdWxkIG1hdGNoIHRoZSBrZXlzIG9mIHRoZSB3ZWlnaHRcbiAgICAvLyBtYW5pZmVzdC5cbiAgICBjb25zdCB3ZWlnaHRWYWx1ZXMgPSBhd2FpdCBpby5sb2FkV2VpZ2h0cyhcbiAgICAgICAgbW9kZWxBbmRXZWlnaHRzQ29uZmlnLndlaWdodHNNYW5pZmVzdCwgbW9kZWxBbmRXZWlnaHRzQ29uZmlnLnBhdGhQcmVmaXgsXG4gICAgICAgIG1vZGVsLndlaWdodHMubWFwKHdlaWdodCA9PiB3ZWlnaHQub3JpZ2luYWxOYW1lKSk7XG5cbiAgICAvLyBNYXAgdGhlIHdlaWdodHMgdG8gdGhlIHVuaXF1ZSB0ZW5zb3IgbmFtZXMgZ2VuZXJhdGVkIGR1cmluZyBtb2RlbCBsb2FkaW5nXG4gICAgY29uc3QgdW5pcXVlV2VpZ2h0VmFsdWVzOiBOYW1lZFRlbnNvck1hcCA9IHt9O1xuICAgIGZvciAoY29uc3Qgd2VpZ2h0IG9mIG1vZGVsLndlaWdodHMpIHtcbiAgICAgIHVuaXF1ZVdlaWdodFZhbHVlc1t3ZWlnaHQub3JpZ2luYWxOYW1lXSA9XG4gICAgICAgICAgd2VpZ2h0VmFsdWVzW3dlaWdodC5vcmlnaW5hbE5hbWVdO1xuICAgIH1cblxuICAgIG1vZGVsLmxvYWRXZWlnaHRzKHVuaXF1ZVdlaWdodFZhbHVlcyk7XG4gICAgLy8gRGlzcG9zZSB0ZW1wb3Jhcnkgd2VpZ2h0IHZhbHVlcy5cbiAgICBkaXNwb3NlKHdlaWdodFZhbHVlcyk7XG4gIH1cbiAgcmV0dXJuIG1vZGVsO1xufVxuXG4vKipcbiAqIE9wdGlvbnMgZm9yIGxvYWRpbmcgYSBzYXZlZCBtb2RlIGluIFRlbnNvckZsb3cuanMgZm9ybWF0LlxuICovXG5leHBvcnQgaW50ZXJmYWNlIE1vZGVsQW5kV2VpZ2h0c0NvbmZpZyB7XG4gIC8qKlxuICAgKiBBIEpTT04gb2JqZWN0IG9yIEpTT04gc3RyaW5nIGNvbnRhaW5pbmcgdGhlIG1vZGVsIGNvbmZpZy5cbiAgICpcbiAgICogVGhpcyBjYW4gYmUgZWl0aGVyIG9mIHRoZSBmb2xsb3dpbmcgdHdvIGZvcm1hdHM6XG4gICAqICAgLSBBIG1vZGVsIGFyY2hpZWN0dXJlLW9ubHkgY29uZmlnLCAgaS5lLiwgYSBmb3JtYXQgY29uc2lzdGVudCB3aXRoIHRoZVxuICAgKiAgICAgcmV0dXJuIHZhbHVlIG9mYGtlcmFzLk1vZGVsLnRvX2pzb24oKWAuXG4gICAqICAgLSBBIGZ1bGwgbW9kZWwgY29uZmlnLCBjb250YWluaW5nIG5vdCBvbmx5IG1vZGVsIGFyY2hpdGVjdHVyZSwgYnV0IGFsc29cbiAgICogICAgIHRyYWluaW5nIG9wdGlvbnMgYW5kIHN0YXRlLCBpLmUuLCBhIGZvcm1hdCBjb25zaXN0ZW50IHdpdGggdGhlIHJldHVyblxuICAgKiAgICAgdmFsdWUgb2YgYGtlcmFzLm1vZGVscy5zYXZlX21vZGVsKClgLlxuICAgKi9cbiAgbW9kZWxUb3BvbG9neTogUHlKc29uRGljdDtcblxuICAvKipcbiAgICogQSB3ZWlnaHRzIG1hbmlmZXN0IGluIFRlbnNvckZsb3cuanMgZm9ybWF0LlxuICAgKi9cbiAgd2VpZ2h0c01hbmlmZXN0PzogaW8uV2VpZ2h0c01hbmlmZXN0Q29uZmlnO1xuXG4gIC8qKlxuICAgKiBQYXRoIHRvIHByZXBlbmQgdG8gdGhlIHBhdGhzIGluIGB3ZWlnaHRNYW5pZmVzdGAgYmVmb3JlIGZldGNoaW5nLlxuICAgKlxuICAgKiBUaGUgcGF0aCBtYXkgb3B0aW9uYWxseSBlbmQgaW4gYSBzbGFzaCAoJy8nKS5cbiAgICovXG4gIHBhdGhQcmVmaXg/OiBzdHJpbmc7XG59XG5cbi8vIFRPRE8obmllbHNlbmUpOiBSZW1vdmUgYWZ0ZXI6IGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzQwMFxuZXhwb3J0IGludGVyZmFjZSBNb2RlbFByZWRpY3RBcmdzIHtcbiAgLyoqXG4gICAqIE9wdGlvbmFsLiBCYXRjaCBzaXplIChJbnRlZ2VyKS4gSWYgdW5zcGVjaWZpZWQsIGl0IHdpbGwgZGVmYXVsdCB0byAzMi5cbiAgICovXG4gIGJhdGNoU2l6ZT86IG51bWJlcjtcblxuICAvKipcbiAgICogT3B0aW9uYWwuIFZlcmJvc2l0eSBtb2RlLiBEZWZhdWx0cyB0byBmYWxzZS5cbiAgICovXG4gIHZlcmJvc2U/OiBib29sZWFuO1xufVxuXG4vKipcbiAqIExvYWQgYSBtb2RlbCBjb21wb3NlZCBvZiBMYXllciBvYmplY3RzLCBpbmNsdWRpbmcgaXRzIHRvcG9sb2d5IGFuZCBvcHRpb25hbGx5XG4gKiB3ZWlnaHRzLiBTZWUgdGhlIFR1dG9yaWFsIG5hbWVkIFwiSG93IHRvIGltcG9ydCBhIEtlcmFzIE1vZGVsXCIgZm9yIHVzYWdlXG4gKiBleGFtcGxlcy5cbiAqXG4gKiBUaGlzIG1ldGhvZCBpcyBhcHBsaWNhYmxlIHRvOlxuICpcbiAqIDEuIE1vZGVscyBjcmVhdGVkIHdpdGggdGhlIGB0Zi5sYXllcnMuKmAsIGB0Zi5zZXF1ZW50aWFsYCwgYW5kXG4gKiBgdGYubW9kZWxgIEFQSXMgb2YgVGVuc29yRmxvdy5qcyBhbmQgbGF0ZXIgc2F2ZWQgd2l0aCB0aGVcbiAqIGB0Zi5MYXllcnNNb2RlbC5zYXZlYCBtZXRob2QuXG4gKiAyLiBNb2RlbHMgY29udmVydGVkIGZyb20gS2VyYXMgb3IgVGVuc29yRmxvdyB0Zi5rZXJhcyB1c2luZyB0aGVcbiAqIFt0ZW5zb3JmbG93anNfY29udmVydGVyXShodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZmpzL3RyZWUvbWFzdGVyL3RmanMtY29udmVydGVyKS5cbiAqXG4gKiBUaGlzIG1vZGUgaXMgKm5vdCogYXBwbGljYWJsZSB0byBUZW5zb3JGbG93IGBTYXZlZE1vZGVsYHMgb3IgdGhlaXIgY29udmVydGVkXG4gKiBmb3Jtcy4gRm9yIHRob3NlIG1vZGVscywgdXNlIGB0Zi5sb2FkR3JhcGhNb2RlbGAuXG4gKlxuICogRXhhbXBsZSAxLiBMb2FkIGEgbW9kZWwgZnJvbSBhbiBIVFRQIHNlcnZlci5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgbW9kZWwgPSBhd2FpdCB0Zi5sb2FkTGF5ZXJzTW9kZWwoXG4gKiAgICAgJ2h0dHBzOi8vc3RvcmFnZS5nb29nbGVhcGlzLmNvbS90ZmpzLW1vZGVscy90ZmpzL2lyaXNfdjEvbW9kZWwuanNvbicpO1xuICogbW9kZWwuc3VtbWFyeSgpO1xuICogYGBgXG4gKlxuICogRXhhbXBsZSAyOiBTYXZlIGBtb2RlbGAncyB0b3BvbG9neSBhbmQgd2VpZ2h0cyB0byBicm93c2VyIFtsb2NhbFxuICogc3RvcmFnZV0oaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL1dpbmRvdy9sb2NhbFN0b3JhZ2UpO1xuICogdGhlbiBsb2FkIGl0IGJhY2suXG4gKlxuICogYGBganNcbiAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbChcbiAqICAgICB7bGF5ZXJzOiBbdGYubGF5ZXJzLmRlbnNlKHt1bml0czogMSwgaW5wdXRTaGFwZTogWzNdfSldfSk7XG4gKiBjb25zb2xlLmxvZygnUHJlZGljdGlvbiBmcm9tIG9yaWdpbmFsIG1vZGVsOicpO1xuICogbW9kZWwucHJlZGljdCh0Zi5vbmVzKFsxLCAzXSkpLnByaW50KCk7XG4gKlxuICogY29uc3Qgc2F2ZVJlc3VsdHMgPSBhd2FpdCBtb2RlbC5zYXZlKCdsb2NhbHN0b3JhZ2U6Ly9teS1tb2RlbC0xJyk7XG4gKlxuICogY29uc3QgbG9hZGVkTW9kZWwgPSBhd2FpdCB0Zi5sb2FkTGF5ZXJzTW9kZWwoJ2xvY2Fsc3RvcmFnZTovL215LW1vZGVsLTEnKTtcbiAqIGNvbnNvbGUubG9nKCdQcmVkaWN0aW9uIGZyb20gbG9hZGVkIG1vZGVsOicpO1xuICogbG9hZGVkTW9kZWwucHJlZGljdCh0Zi5vbmVzKFsxLCAzXSkpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBFeGFtcGxlIDMuIFNhdmluZyBgbW9kZWxgJ3MgdG9wb2xvZ3kgYW5kIHdlaWdodHMgdG8gYnJvd3NlclxuICogW0luZGV4ZWREQl0oaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL0luZGV4ZWREQl9BUEkpO1xuICogdGhlbiBsb2FkIGl0IGJhY2suXG4gKlxuICogYGBganNcbiAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbChcbiAqICAgICB7bGF5ZXJzOiBbdGYubGF5ZXJzLmRlbnNlKHt1bml0czogMSwgaW5wdXRTaGFwZTogWzNdfSldfSk7XG4gKiBjb25zb2xlLmxvZygnUHJlZGljdGlvbiBmcm9tIG9yaWdpbmFsIG1vZGVsOicpO1xuICogbW9kZWwucHJlZGljdCh0Zi5vbmVzKFsxLCAzXSkpLnByaW50KCk7XG4gKlxuICogY29uc3Qgc2F2ZVJlc3VsdHMgPSBhd2FpdCBtb2RlbC5zYXZlKCdpbmRleGVkZGI6Ly9teS1tb2RlbC0xJyk7XG4gKlxuICogY29uc3QgbG9hZGVkTW9kZWwgPSBhd2FpdCB0Zi5sb2FkTGF5ZXJzTW9kZWwoJ2luZGV4ZWRkYjovL215LW1vZGVsLTEnKTtcbiAqIGNvbnNvbGUubG9nKCdQcmVkaWN0aW9uIGZyb20gbG9hZGVkIG1vZGVsOicpO1xuICogbG9hZGVkTW9kZWwucHJlZGljdCh0Zi5vbmVzKFsxLCAzXSkpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBFeGFtcGxlIDQuIExvYWQgYSBtb2RlbCBmcm9tIHVzZXItc2VsZWN0ZWQgZmlsZXMgZnJvbSBIVE1MXG4gKiBbZmlsZSBpbnB1dFxuICogZWxlbWVudHNdKGh0dHBzOi8vZGV2ZWxvcGVyLm1vemlsbGEub3JnL2VuLVVTL2RvY3MvV2ViL0hUTUwvRWxlbWVudC9pbnB1dC9maWxlKS5cbiAqXG4gKiBgYGBqc1xuICogLy8gTm90ZTogdGhpcyBjb2RlIHNuaXBwZXQgd2lsbCBub3Qgd29yayB3aXRob3V0IHRoZSBIVE1MIGVsZW1lbnRzIGluIHRoZVxuICogLy8gICBwYWdlXG4gKiBjb25zdCBqc29uVXBsb2FkID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2pzb24tdXBsb2FkJyk7XG4gKiBjb25zdCB3ZWlnaHRzVXBsb2FkID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3dlaWdodHMtdXBsb2FkJyk7XG4gKlxuICogY29uc3QgbW9kZWwgPSBhd2FpdCB0Zi5sb2FkTGF5ZXJzTW9kZWwoXG4gKiAgICAgdGYuaW8uYnJvd3NlckZpbGVzKFtqc29uVXBsb2FkLmZpbGVzWzBdLCB3ZWlnaHRzVXBsb2FkLmZpbGVzWzBdXSkpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHBhdGhPcklPSGFuZGxlciBDYW4gYmUgZWl0aGVyIG9mIHRoZSB0d28gZm9ybWF0c1xuICogICAxLiBBIHN0cmluZyBwYXRoIHRvIHRoZSBgTW9kZWxBbmRXZWlnaHRzQ29uZmlnYCBKU09OIGRlc2NyaWJpbmdcbiAqICAgICAgdGhlIG1vZGVsIGluIHRoZSBjYW5vbmljYWwgVGVuc29yRmxvdy5qcyBmb3JtYXQuIEZvciBmaWxlOi8vXG4gKiAgICAgICh0ZmpzLW5vZGUtb25seSksIGh0dHA6Ly8gYW5kIGh0dHBzOi8vIHNjaGVtYXMsIHRoZSBwYXRoIGNhbiBiZVxuICogICAgICBlaXRoZXIgYWJzb2x1dGUgb3IgcmVsYXRpdmUuIFRoZSBjb250ZW50IG9mIHRoZSBKU09OIGZpbGUgaXMgYXNzdW1lZCB0b1xuICogICAgICBiZSBhIEpTT04gb2JqZWN0IHdpdGggdGhlIGZvbGxvd2luZyBmaWVsZHMgYW5kIHZhbHVlczpcbiAqICAgICAgLSAnbW9kZWxUb3BvbG9neSc6IEEgSlNPTiBvYmplY3QgdGhhdCBjYW4gYmUgZWl0aGVyIG9mOlxuICogICAgICAgIDEuIGEgbW9kZWwgYXJjaGl0ZWN0dXJlIEpTT04gY29uc2lzdGVudCB3aXRoIHRoZSBmb3JtYXQgb2YgdGhlIHJldHVyblxuICogICAgICAgICAgICB2YWx1ZSBvZiBga2VyYXMuTW9kZWwudG9fanNvbigpYFxuICogICAgICAgIDIuIGEgZnVsbCBtb2RlbCBKU09OIGluIHRoZSBmb3JtYXQgb2YgYGtlcmFzLm1vZGVscy5zYXZlX21vZGVsKClgLlxuICogICAgICAtICd3ZWlnaHRzTWFuaWZlc3QnOiBBIFRlbnNvckZsb3cuanMgd2VpZ2h0cyBtYW5pZmVzdC5cbiAqICAgICAgU2VlIHRoZSBQeXRob24gY29udmVydGVyIGZ1bmN0aW9uIGBzYXZlX21vZGVsKClgIGZvciBtb3JlIGRldGFpbHMuXG4gKiAgICAgIEl0IGlzIGFsc28gYXNzdW1lZCB0aGF0IG1vZGVsIHdlaWdodHMgY2FuIGJlIGFjY2Vzc2VkIGZyb20gcmVsYXRpdmVcbiAqICAgICAgcGF0aHMgZGVzY3JpYmVkIGJ5IHRoZSBgcGF0aHNgIGZpZWxkcyBpbiB3ZWlnaHRzIG1hbmlmZXN0LlxuICogICAyLiBBIGB0Zi5pby5JT0hhbmRsZXJgIG9iamVjdCB0aGF0IGxvYWRzIG1vZGVsIGFydGlmYWN0cyB3aXRoIGl0cyBgbG9hZGBcbiAqICAgICAgbWV0aG9kLlxuICogQHBhcmFtIG9wdGlvbnMgT3B0aW9uYWwgY29uZmlndXJhdGlvbiBhcmd1bWVudHMgZm9yIHRoZSBtb2RlbCBsb2FkaW5nLFxuICogICBpbmNsdWRpbmc6XG4gKiAgIC0gYHN0cmljdGA6IFJlcXVpcmUgdGhhdCB0aGUgcHJvdmlkZWQgd2VpZ2h0cyBleGFjdGx5IG1hdGNoIHRob3NlIHJlcXVpcmVkXG4gKiAgICAgYnkgdGhlIGxheWVycy4gIERlZmF1bHQgdHJ1ZS4gIFBhc3NpbmcgZmFsc2UgbWVhbnMgdGhhdCBib3RoIGV4dHJhXG4gKiAgICAgd2VpZ2h0cyBhbmQgbWlzc2luZyB3ZWlnaHRzIHdpbGwgYmUgc2lsZW50bHkgaWdub3JlZC5cbiAqICAgLSBgb25Qcm9ncmVzc2A6IEEgcHJvZ3Jlc3MgY2FsbGJhY2sgb2YgdGhlIGZvcm06XG4gKiAgICAgYChmcmFjdGlvbjogbnVtYmVyKSA9PiB2b2lkYC4gVGhpcyBjYWxsYmFjayBjYW4gYmUgdXNlZCB0byBtb25pdG9yIHRoZVxuICogICAgIG1vZGVsLWxvYWRpbmcgcHJvY2Vzcy5cbiAqIEByZXR1cm5zIEEgYFByb21pc2VgIG9mIGB0Zi5MYXllcnNNb2RlbGAsIHdpdGggdGhlIHRvcG9sb2d5IGFuZCB3ZWlnaHRzXG4gKiAgICAgbG9hZGVkLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnTG9hZGluZyd9XG4gKi9cbmV4cG9ydCBhc3luYyBmdW5jdGlvbiBsb2FkTGF5ZXJzTW9kZWwoXG4gICAgcGF0aE9ySU9IYW5kbGVyOiBzdHJpbmd8aW8uSU9IYW5kbGVyLFxuICAgIG9wdGlvbnM/OiBpby5Mb2FkT3B0aW9ucyk6IFByb21pc2U8TGF5ZXJzTW9kZWw+IHtcbiAgaWYgKG9wdGlvbnMgPT0gbnVsbCkge1xuICAgIG9wdGlvbnMgPSB7fTtcbiAgfVxuICBpZiAodHlwZW9mIHBhdGhPcklPSGFuZGxlciA9PT0gJ3N0cmluZycpIHtcbiAgICBjb25zdCBoYW5kbGVycyA9IGlvLmdldExvYWRIYW5kbGVycyhwYXRoT3JJT0hhbmRsZXIsIG9wdGlvbnMpO1xuICAgIGlmIChoYW5kbGVycy5sZW5ndGggPT09IDApIHtcbiAgICAgIC8vIEZvciBiYWNrd2FyZCBjb21wYXRpYmlsaXR5OiBpZiBubyBsb2FkIGhhbmRsZXIgY2FuIGJlIGZvdW5kLFxuICAgICAgLy8gYXNzdW1lIGl0IGlzIGEgcmVsYXRpdmUgaHR0cCBwYXRoLlxuICAgICAgLy8gVE9ETyhjYWlzKTogUmVmb3JtYXQgdGhlIGFyZ3MgaW50byBhIHNpbmdsZSBgTG9hZE9wdGlvbnNgIG9uY2UgdGhlIGNvcmVcbiAgICAgIC8vIGlzIHJlZmFjdG9yZWQuXG4gICAgICBoYW5kbGVycy5wdXNoKGlvLmJyb3dzZXJIVFRQUmVxdWVzdChwYXRoT3JJT0hhbmRsZXIsIG9wdGlvbnMpKTtcbiAgICB9IGVsc2UgaWYgKGhhbmRsZXJzLmxlbmd0aCA+IDEpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBGb3VuZCBtb3JlIHRoYW4gb25lICgke2hhbmRsZXJzLmxlbmd0aH0pIGxvYWQgaGFuZGxlcnMgZm9yIGAgK1xuICAgICAgICAgIGBVUkwgJyR7cGF0aE9ySU9IYW5kbGVyfSdgKTtcbiAgICB9XG4gICAgcGF0aE9ySU9IYW5kbGVyID0gaGFuZGxlcnNbMF07XG4gIH1cbiAgcmV0dXJuIGxvYWRMYXllcnNNb2RlbEZyb21JT0hhbmRsZXIocGF0aE9ySU9IYW5kbGVyLCB1bmRlZmluZWQsIG9wdGlvbnMpO1xufVxuXG4vKipcbiAqIExvYWQgYSBtb2RlbCBhbmQgb3B0aW9uYWxseSBpdHMgd2VpZ2h0cywgdXNpbmcgYW4gSU9IYW5kbGVyIG9iamVjdC5cbiAqXG4gKiBAcGFyYW0gaGFuZGxlciBUaGUgaW5zdGFuY2Ugb2YgYElPSGFuZGxlcmAgdG8gYmUgdXNlZCBkdXJpbmcgdGhlIG1vZGVsXG4gKiAgIGxvYWRpbmcuXG4gKiBAcGFyYW0gY3VzdG9tT2JqZWN0cyBBbnkgb3B0aW9uYWwgY3VzdG9tIG9iamVjdHMgdG8gYmUgdXNlZCBkdXJpbmcgbW9kZWxcbiAqICAgbG9hZGluZy5cbiAqIEBwYXJhbSBzdHJpY3QgV2hldGhlciB0aGUgd2VpZ2h0IGxvYWRpbmcgd2lsbCBiZSBkb25lIGluIHN0cmljdCBtb2RlLlxuICogICBEZWZhdWx0OiBgdHJ1ZWAuXG4gKi9cbmV4cG9ydCBhc3luYyBmdW5jdGlvbiBsb2FkTGF5ZXJzTW9kZWxGcm9tSU9IYW5kbGVyKFxuICAgIGhhbmRsZXI6IGlvLklPSGFuZGxlciwgY3VzdG9tT2JqZWN0cz86IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCxcbiAgICBvcHRpb25zPzogaW8uTG9hZE9wdGlvbnMpOiBQcm9taXNlPExheWVyc01vZGVsPiB7XG4gIGlmIChvcHRpb25zID09IG51bGwpIHtcbiAgICBvcHRpb25zID0ge307XG4gIH1cbiAgaWYgKGhhbmRsZXIubG9hZCA9PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICdDYW5ub3QgcHJvY2VlZCB3aXRoIG1vZGVsIGxvYWRpbmcgYmVjYXVzZSB0aGUgSU9IYW5kbGVyIHByb3ZpZGVkICcgK1xuICAgICAgICAnZG9lcyBub3QgaGF2ZSB0aGUgYGxvYWRgIG1ldGhvZCBpbXBsZW1lbnRlZC4nKTtcbiAgfVxuICBjb25zdCBhcnRpZmFjdHMgPSBhd2FpdCBoYW5kbGVyLmxvYWQoKTtcbiAgbGV0IG1vZGVsVG9wb2xvZ3kgPSBhcnRpZmFjdHMubW9kZWxUb3BvbG9neSBhcyBQeUpzb25EaWN0O1xuICBpZiAobW9kZWxUb3BvbG9neVsnbW9kZWxfY29uZmlnJ10gIT0gbnVsbCkge1xuICAgIG1vZGVsVG9wb2xvZ3kgPSBtb2RlbFRvcG9sb2d5Wydtb2RlbF9jb25maWcnXSBhcyBQeUpzb25EaWN0O1xuICB9XG5cbiAgY29uc3Qgc3RyaWN0ID0gb3B0aW9ucy5zdHJpY3QgPT0gbnVsbCA/IHRydWUgOiBvcHRpb25zLnN0cmljdDtcbiAgLy8gSWYgd2VpZ2h0cyBhcmUgcHJvdmlkZWQgYW5kIHRoZSB3ZWlnaHQtbG9hZGluZyBtb2RlIGlzIHN0cmljdCwgdXNlXG4gIC8vIGZhc3Qgd2VpZ2h0IGluaXRpYWxpemF0aW9uLiBUaGlzIHNraXBzIGNvc3RseSBpbml0aWFsaXplcnMgc3VjaCBhc1xuICAvLyAnb3J0aG9nb25hbCcgYW5kIHNhdmVzIHVubmVjZXNzYXJ5IGNvbXB1dGF0aW9uIGluIGNhc2VzIHdoZXJlXG4gIC8vIHRoZSBpbml0aWFsaXplZCB3ZWlnaHQgdmFsdWVzIHdpbGwgaW1tZWRpYXRlbHkgYmUgb3ZlcndyaXR0ZW4gYnlcbiAgLy8gbG9hZGVkIHdlaWdodCB2YWx1ZXMuXG4gIGNvbnN0IGZhc3RXZWlnaHRJbml0ID1cbiAgICAgIGFydGlmYWN0cy53ZWlnaHREYXRhICE9IG51bGwgJiYgYXJ0aWZhY3RzLndlaWdodFNwZWNzICE9IG51bGwgJiYgc3RyaWN0O1xuICBjb25zdCBtb2RlbCA9XG4gICAgICBkZXNlcmlhbGl6ZShcbiAgICAgICAgICBjb252ZXJ0UHl0aG9uaWNUb1RzKG1vZGVsVG9wb2xvZ3kpIGFzIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCxcbiAgICAgICAgICBjdXN0b21PYmplY3RzLCBmYXN0V2VpZ2h0SW5pdCkgYXMgTGF5ZXJzTW9kZWw7XG5cbiAgY29uc3QgdHJhaW5pbmdDb25maWcgPSBhcnRpZmFjdHMudHJhaW5pbmdDb25maWcgYXMgVHJhaW5pbmdDb25maWc7XG4gIGlmICh0cmFpbmluZ0NvbmZpZyAhPSBudWxsKSB7XG4gICAgbW9kZWwubG9hZFRyYWluaW5nQ29uZmlnKHRyYWluaW5nQ29uZmlnKTtcbiAgfVxuICBpZiAoYXJ0aWZhY3RzLnVzZXJEZWZpbmVkTWV0YWRhdGEgIT0gbnVsbCkge1xuICAgIG1vZGVsLnNldFVzZXJEZWZpbmVkTWV0YWRhdGEoYXJ0aWZhY3RzLnVzZXJEZWZpbmVkTWV0YWRhdGEpO1xuICB9XG5cbiAgLy8gSWYgd2VpZ2h0RGF0YSBpcyBwcmVzZW50LCBsb2FkIHRoZSB3ZWlnaHRzIGludG8gdGhlIG1vZGVsLlxuICBpZiAoYXJ0aWZhY3RzLndlaWdodERhdGEgIT0gbnVsbCkge1xuICAgIC8vIExvYWRpbmcgd2VpZ2h0cyByZXF1aXJlcyB3ZWlnaHRTcGVjcy5cbiAgICBpZiAoYXJ0aWZhY3RzLndlaWdodFNwZWNzID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdMYXllcnNNb2RlbCBhcnRpZmFjdHMgY29udGFpbnMgd2VpZ2h0IGRhdGEsIGJ1dCBub3Qgd2VpZ2h0IHNwZWNzLiAnICtcbiAgICAgICAgICAnVGhlcmVmb3JlIGxvYWRpbmcgb2Ygd2VpZ2h0cyBjYW5ub3QgcHJvY2VlZC4nKTtcbiAgICB9XG5cbiAgICBjb25zdCB7bW9kZWxXZWlnaHRzLCBvcHRpbWl6ZXJXZWlnaHRzfSA9IGRlY29kZU1vZGVsQW5kT3B0aW1pemVyV2VpZ2h0cyhcbiAgICAgICAgYXJ0aWZhY3RzLndlaWdodERhdGEsIGFydGlmYWN0cy53ZWlnaHRTcGVjcyk7XG4gICAgbW9kZWwubG9hZFdlaWdodHMobW9kZWxXZWlnaHRzLCBzdHJpY3QpO1xuXG4gICAgaWYgKG1vZGVsLm9wdGltaXplciAhPSBudWxsICYmIG9wdGltaXplcldlaWdodHMubGVuZ3RoID4gMCkge1xuICAgICAgYXdhaXQgbW9kZWwub3B0aW1pemVyLnNldFdlaWdodHMob3B0aW1pemVyV2VpZ2h0cyk7XG4gICAgfVxuXG4gICAgLy8gRGlzcG9zZSB0ZW1wb3Jhcnkgd2VpZ2h0IHZhbHVlcy5cbiAgICBkaXNwb3NlKG1vZGVsV2VpZ2h0cyk7XG4gICAgZGlzcG9zZShvcHRpbWl6ZXJXZWlnaHRzLm1hcCh3ID0+IHcudGVuc29yKSk7XG4gIH1cbiAgcmV0dXJuIG1vZGVsO1xufVxuXG5mdW5jdGlvbiBkZWNvZGVNb2RlbEFuZE9wdGltaXplcldlaWdodHMoXG4gICAgYnVmZmVyOiBBcnJheUJ1ZmZlciwgc3BlY3M6IGlvLldlaWdodHNNYW5pZmVzdEVudHJ5W10pOlxuICAgIHttb2RlbFdlaWdodHM6IE5hbWVkVGVuc29yTWFwLCBvcHRpbWl6ZXJXZWlnaHRzOiBOYW1lZFRlbnNvcltdfSB7XG4gIGNvbnN0IG5hbWUyVGVuc29yID0gaW8uZGVjb2RlV2VpZ2h0cyhidWZmZXIsIHNwZWNzKTtcbiAgY29uc3QgbW9kZWxXZWlnaHRzOiBOYW1lZFRlbnNvck1hcCA9IHt9O1xuICBjb25zdCBvcHRpbWl6ZXJXZWlnaHRzOiBOYW1lZFRlbnNvcltdID0gW107XG4gIHNwZWNzLmZvckVhY2goc3BlYyA9PiB7XG4gICAgaWYgKHNwZWMuZ3JvdXAgPT09ICdvcHRpbWl6ZXInKSB7XG4gICAgICBvcHRpbWl6ZXJXZWlnaHRzLnB1c2goe25hbWU6IHNwZWMubmFtZSwgdGVuc29yOiBuYW1lMlRlbnNvcltzcGVjLm5hbWVdfSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIG1vZGVsV2VpZ2h0c1tzcGVjLm5hbWVdID0gbmFtZTJUZW5zb3Jbc3BlYy5uYW1lXTtcbiAgICB9XG4gIH0pO1xuICByZXR1cm4ge21vZGVsV2VpZ2h0cywgb3B0aW1pemVyV2VpZ2h0c307XG59XG5cbi8qKlxuICogQ29uZmlndXJhdGlvbiBmb3IgYSBTZXF1ZW50aWFsIG1vZGVsLlxuICovXG5leHBvcnQgaW50ZXJmYWNlIFNlcXVlbnRpYWxBcmdzIHtcbiAgLyoqIFN0YWNrIG9mIGxheWVycyBmb3IgdGhlIG1vZGVsLiAqL1xuICBsYXllcnM/OiBMYXllcltdO1xuXG4gIC8qKiBUaGUgbmFtZSBvZiB0aGlzIG1vZGVsLiAqL1xuICBuYW1lPzogc3RyaW5nO1xufVxuXG4vKipcbiAqIEEgbW9kZWwgd2l0aCBhIHN0YWNrIG9mIGxheWVycywgZmVlZGluZyBsaW5lYXJseSBmcm9tIG9uZSB0byB0aGUgbmV4dC5cbiAqXG4gKiBgdGYuc2VxdWVudGlhbGAgaXMgYSBmYWN0b3J5IGZ1bmN0aW9uIHRoYXQgY3JlYXRlcyBhbiBpbnN0YW5jZSBvZlxuICogYHRmLlNlcXVlbnRpYWxgLlxuICpcbiAqIGBgYGpzXG4gKiAgLy8gRGVmaW5lIGEgbW9kZWwgZm9yIGxpbmVhciByZWdyZXNzaW9uLlxuICogIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCgpO1xuICogIG1vZGVsLmFkZCh0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbMV19KSk7XG4gKlxuICogIC8vIFByZXBhcmUgdGhlIG1vZGVsIGZvciB0cmFpbmluZzogU3BlY2lmeSB0aGUgbG9zcyBhbmQgdGhlIG9wdGltaXplci5cbiAqICBtb2RlbC5jb21waWxlKHtsb3NzOiAnbWVhblNxdWFyZWRFcnJvcicsIG9wdGltaXplcjogJ3NnZCd9KTtcbiAqXG4gKiAgLy8gR2VuZXJhdGUgc29tZSBzeW50aGV0aWMgZGF0YSBmb3IgdHJhaW5pbmcuXG4gKiAgY29uc3QgeHMgPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgNF0sIFs0LCAxXSk7XG4gKiAgY29uc3QgeXMgPSB0Zi50ZW5zb3IyZChbMSwgMywgNSwgN10sIFs0LCAxXSk7XG4gKlxuICogIC8vIFRyYWluIHRoZSBtb2RlbCB1c2luZyB0aGUgZGF0YSB0aGVuIGRvIGluZmVyZW5jZSBvbiBhIGRhdGEgcG9pbnQgdGhlXG4gKiAgLy8gbW9kZWwgaGFzbid0IHNlZW46XG4gKiAgYXdhaXQgbW9kZWwuZml0KHhzLCB5cyk7XG4gKiAgbW9kZWwucHJlZGljdCh0Zi50ZW5zb3IyZChbNV0sIFsxLCAxXSkpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICovXG5leHBvcnQgY2xhc3MgU2VxdWVudGlhbCBleHRlbmRzIExheWVyc01vZGVsIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBjbGFzc05hbWUgPSAnU2VxdWVudGlhbCc7XG4gIHByaXZhdGUgbW9kZWw6IExheWVyc01vZGVsO1xuICBjb25zdHJ1Y3RvcihhcmdzPzogU2VxdWVudGlhbEFyZ3MpIHtcbiAgICBzdXBlcih7aW5wdXRzOiBbXSwgb3V0cHV0czogW119KTtcbiAgICBhcmdzID0gYXJncyB8fCB7fTtcblxuICAgIHRoaXMudHJhaW5hYmxlID0gdHJ1ZTtcbiAgICB0aGlzLmJ1aWx0ID0gZmFsc2U7XG5cbiAgICAvLyBTZXQgbW9kZWwgbmFtZS5cbiAgICB0aGlzLm5hbWUgPSAoYXJncy5uYW1lICE9IG51bGwpID8gYXJncy5uYW1lIDogZ2V0VWlkKCdzZXF1ZW50aWFsXycpO1xuXG4gICAgLy8gQWRkIHRvIHRoZSBtb2RlbCBhbnkgbGF5ZXJzIHBhc3NlZCB0byB0aGUgY29uc3RydWN0b3IuXG4gICAgaWYgKGFyZ3MubGF5ZXJzICE9IG51bGwpIHtcbiAgICAgIGZvciAoY29uc3QgbGF5ZXIgb2YgYXJncy5sYXllcnMpIHtcbiAgICAgICAgdGhpcy5hZGQobGF5ZXIpO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIC8vIEhlbHBlciBmdW5jdGlvbiB0byBTZXF1ZW50aWFsLmFkZCAgVGhyb3dzIGlmIHRoZSBuZXcgb3V0cHV0IHNoYXBlIHdpbGwgYmVcbiAgLy8gaW52YWxpZC5cbiAgcHJpdmF0ZSBjaGVja1NoYXBlKGxheWVyOiBMYXllcikge1xuICAgIGNvbnN0IHNoYXBlID0gbGF5ZXIuaW5ib3VuZE5vZGVzWzBdLm91dHB1dFRlbnNvcnNbMF0uc2hhcGU7XG4gICAgaWYgKHNoYXBlLnNvbWUoeCA9PiB4IDwgMCkpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdOZWdhdGl2ZSBkaW1lbnNpb24gc2l6ZSBjYXVzZWQgYnkgYWRkaW5nIGxheWVyICcgK1xuICAgICAgICAgIGAke2xheWVyLm5hbWV9IHdpdGggaW5wdXQgc2hhcGUgW2AgK1xuICAgICAgICAgIGAke2xheWVyLmluYm91bmROb2Rlc1swXS5pbnB1dFRlbnNvcnNbMF0uc2hhcGV9XWApO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBBZGRzIGEgbGF5ZXIgaW5zdGFuY2Ugb24gdG9wIG9mIHRoZSBsYXllciBzdGFjay5cbiAgICpcbiAgICogYGBganNcbiAgICogIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCgpO1xuICAgKiAgbW9kZWwuYWRkKHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDgsIGlucHV0U2hhcGU6IFsxXX0pKTtcbiAgICogIG1vZGVsLmFkZCh0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiA0LCBhY3RpdmF0aW9uOiAncmVsdTYnfSkpO1xuICAgKiAgbW9kZWwuYWRkKHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGFjdGl2YXRpb246ICdyZWx1Nid9KSk7XG4gICAqICAvLyBOb3RlIHRoYXQgdGhlIHVudHJhaW5lZCBtb2RlbCBpcyByYW5kb20gYXQgdGhpcyBwb2ludC5cbiAgICogIG1vZGVsLnByZWRpY3QodGYucmFuZG9tTm9ybWFsKFsxMCwgMV0pKS5wcmludCgpO1xuICAgKiBgYGBcbiAgICogQHBhcmFtIGxheWVyIExheWVyIGluc3RhbmNlLlxuICAgKlxuICAgKiBAZXhjZXB0aW9uIFZhbHVlRXJyb3IgSW4gY2FzZSB0aGUgYGxheWVyYCBhcmd1bWVudCBkb2VzIG5vdCBrbm93IGl0c1xuICAgKiBpbnB1dCBzaGFwZS5cbiAgICogQGV4Y2VwdGlvbiBWYWx1ZUVycm9yIEluIGNhc2UgdGhlIGBsYXllcmAgYXJndW1lbnQgaGFzIG11bHRpcGxlIG91dHB1dFxuICAgKiAgIHRlbnNvcnMsIG9yIGlzIGFscmVhZHkgY29ubmVjdGVkIHNvbWV3aGVyZSBlbHNlIChmb3JiaWRkZW4gaW5cbiAgICogICBgU2VxdWVudGlhbGAgbW9kZWxzKS5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFkZChsYXllcjogTGF5ZXIpOiB2b2lkIHtcbiAgICBjb25zdCBpc0xheWVyTW9kZWxJbnN0YW5jZSA9XG4gICAgICAgIGxheWVyIGluc3RhbmNlb2YgU2VxdWVudGlhbCB8fCBsYXllciBpbnN0YW5jZW9mIExheWVyc01vZGVsO1xuICAgIGxldCBtb2RlbExheWVyOiBMYXllcnNNb2RlbDtcbiAgICBpZiAoaXNMYXllck1vZGVsSW5zdGFuY2UpIHtcbiAgICAgIG1vZGVsTGF5ZXIgPSBsYXllciBhcyBMYXllcnNNb2RlbDtcbiAgICAgIGlmIChtb2RlbExheWVyLm91dHB1dHMubGVuZ3RoICE9PSAxKSB7XG4gICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgJ0FsbCBsYXllcnMgaW4gYSBTZXF1ZW50aWFsIG1vZGVsICcgK1xuICAgICAgICAgICAgJ3Nob3VsZCBoYXZlIGEgc2luZ2xlIG91dHB1dCB0ZW5zb3IuICcgK1xuICAgICAgICAgICAgJ0ZvciBtdWx0aS1vdXRwdXQgbGF5ZXJzLCAnICtcbiAgICAgICAgICAgICd1c2UgdGhlIGZ1bmN0aW9uYWwgQVBJLicpO1xuICAgICAgfVxuICAgICAgaWYgKG1vZGVsTGF5ZXIuaW5wdXRzLmxlbmd0aCAhPT0gMSkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICdBbGwgbGF5ZXJzIGluIGEgU2VxdWVudGlhbCBtb2RlbCAnICtcbiAgICAgICAgICAgICdzaG91bGQgaGF2ZSBhIHNpbmdsZSBpbnB1dCB0ZW5zb3IuICcgK1xuICAgICAgICAgICAgJ0ZvciBtdWx0aS1pbnB1dCBsYXllcnMsICcgK1xuICAgICAgICAgICAgJ3VzZSB0aGUgZnVuY3Rpb25hbCBBUEkuJyk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKHRoaXMub3V0cHV0cy5sZW5ndGggPT09IDApIHtcbiAgICAgIC8vIGZpcnN0IGxheWVyIGluIG1vZGVsOiBjaGVjayB0aGF0IGl0IGlzIGFuIGlucHV0IGxheWVyXG4gICAgICBpZiAobGF5ZXIuaW5ib3VuZE5vZGVzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgICAvLyBjcmVhdGUgYW4gaW5wdXQgbGF5ZXJcbiAgICAgICAgaWYgKGxheWVyLmJhdGNoSW5wdXRTaGFwZSA9PSBudWxsKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgICdUaGUgZmlyc3QgbGF5ZXIgaW4gYSBTZXF1ZW50aWFsIG1vZGVsIG11c3QgJyArXG4gICAgICAgICAgICAgICdnZXQgYW4gYGlucHV0U2hhcGVgIG9yIGBiYXRjaElucHV0U2hhcGVgIGFyZ3VtZW50LicpO1xuICAgICAgICB9XG4gICAgICAgIC8vIEluc3RhbnRpYXRlIHRoZSBpbnB1dCBsYXllci5cbiAgICAgICAgY29uc3QgeCA9IElucHV0KHtcbiAgICAgICAgICBiYXRjaFNoYXBlOiBsYXllci5iYXRjaElucHV0U2hhcGUsXG4gICAgICAgICAgZHR5cGU6IGxheWVyLmR0eXBlLFxuICAgICAgICAgIG5hbWU6IGxheWVyLm5hbWUgKyAnX2lucHV0J1xuICAgICAgICB9KTtcbiAgICAgICAgLy8gVGhpcyB3aWxsIGJ1aWxkIHRoZSBjdXJyZW50IGxheWVyIGFuZCBjcmVhdGUgdGhlIG5vZGUgY29ubmVjdGluZ1xuICAgICAgICAvLyB0aGUgY3VycmVudCBsYXllciB0byB0aGUgaW5wdXQgbGF5ZXIgd2UganVzdCBjcmVhdGVkLlxuICAgICAgICBsYXllci5hcHBseSh4KTtcbiAgICAgIH1cblxuICAgICAgaWYgKGlzTGF5ZXJNb2RlbEluc3RhbmNlKSB7XG4gICAgICAgIHRoaXMub3V0cHV0cyA9IG1vZGVsTGF5ZXIub3V0cHV0cztcbiAgICAgICAgdGhpcy5pbnB1dHMgPSBtb2RlbExheWVyLmlucHV0cztcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGlmIChsYXllci5pbmJvdW5kTm9kZXMubGVuZ3RoICE9PSAxKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgICdBIGxheWVyIGFkZGVkIHRvIGEgU2VxdWVudGlhbCBtb2RlbCBtdXN0IG5vdCBhbHJlYWR5IGJlICcgK1xuICAgICAgICAgICAgICBgY29ubmVjdGVkIHNvbWV3aGVyZSBlbHNlLiBMYXllcnNNb2RlbCByZWNlaXZlZCBsYXllciAke1xuICAgICAgICAgICAgICAgICAgbGF5ZXIubmFtZX0gYCArXG4gICAgICAgICAgICAgIGB3aGljaCBoYXMgJHtsYXllci5pbmJvdW5kTm9kZXMubGVuZ3RofSBwcmUtZXhpc3RpbmcgaW5ib3VuZCBgICtcbiAgICAgICAgICAgICAgJ2Nvbm5lY3Rpb25zLicpO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYgKGxheWVyLmluYm91bmROb2Rlc1swXS5vdXRwdXRUZW5zb3JzLmxlbmd0aCAhPT0gMSkge1xuICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgICAnQWxsIGxheWVycyBpbiBhIFNlcXVlbnRpYWwgbW9kZWwgJyArXG4gICAgICAgICAgICAgICdzaG91bGQgaGF2ZSBhIHNpbmdsZSBvdXRwdXQgdGVuc29yLiAnICtcbiAgICAgICAgICAgICAgJ0ZvciBtdWx0aS1vdXRwdXQgbGF5ZXJzLCAnICtcbiAgICAgICAgICAgICAgJ3VzZSB0aGUgZnVuY3Rpb25hbCBBUEkuJyk7XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy5jaGVja1NoYXBlKGxheWVyKTtcbiAgICAgICAgdGhpcy5vdXRwdXRzID0gW2xheWVyLmluYm91bmROb2Rlc1swXS5vdXRwdXRUZW5zb3JzWzBdXTtcbiAgICAgICAgdGhpcy5pbnB1dHMgPSBnZXRTb3VyY2VJbnB1dHModGhpcy5vdXRwdXRzWzBdKTtcbiAgICAgIH1cblxuICAgICAgdGhpcy5pbmJvdW5kTm9kZXMgPSBbXTtcbiAgICAgIC8vIFdlIGNyZWF0ZSBhbiBpbnB1dCBub2RlLCB3aGljaCB3ZSB3aWxsIGtlZXAgdXBkYXRlZFxuICAgICAgLy8gYXMgd2UgYWRkIG1vcmUgbGF5ZXJzLlxuICAgICAgLy8gKFRoaXMgY2FsbCBoYXMgc2lkZSBlZmZlY3RzLilcbiAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby11bnVzZWQtZXhwcmVzc2lvblxuICAgICAgbmV3IE5vZGUoe1xuICAgICAgICBvdXRib3VuZExheWVyOiB0aGlzLFxuICAgICAgICBpbmJvdW5kTGF5ZXJzOiBbXSxcbiAgICAgICAgbm9kZUluZGljZXM6IFtdLFxuICAgICAgICB0ZW5zb3JJbmRpY2VzOiBbXSxcbiAgICAgICAgaW5wdXRUZW5zb3JzOiB0aGlzLmlucHV0cyxcbiAgICAgICAgb3V0cHV0VGVuc29yczogdGhpcy5vdXRwdXRzLFxuICAgICAgICAvLyBubyBtb2RlbC1sZXZlbCBtYXNraW5nIGZvciBub3dcbiAgICAgICAgaW5wdXRNYXNrczogZ2VuZXJpY191dGlscy5weUxpc3RSZXBlYXQobnVsbCwgdGhpcy5pbnB1dHMubGVuZ3RoKSxcbiAgICAgICAgb3V0cHV0TWFza3M6IFtudWxsXSxcbiAgICAgICAgaW5wdXRTaGFwZXM6IHRoaXMuaW5wdXRzLm1hcCh4ID0+IHguc2hhcGUpLFxuICAgICAgICBvdXRwdXRTaGFwZXM6IHRoaXMub3V0cHV0c1swXS5zaGFwZVxuICAgICAgfSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IG91dHB1dFRlbnNvciA9IGxheWVyLmFwcGx5KHRoaXMub3V0cHV0c1swXSk7XG4gICAgICBpZiAoQXJyYXkuaXNBcnJheShvdXRwdXRUZW5zb3IpKSB7XG4gICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoXG4gICAgICAgICAgICAnQWxsIGxheWVycyBpbiBhIFNlcXVlbnRpYWwgbW9kZWwgJyArXG4gICAgICAgICAgICAnc2hvdWxkIGhhdmUgYSBzaW5nbGUgb3V0cHV0IHRlbnNvci4gJyArXG4gICAgICAgICAgICAnRm9yIG11bHRpLW91dHB1dCBsYXllcnMsICcgK1xuICAgICAgICAgICAgJ3VzZSB0aGUgZnVuY3Rpb25hbCBBUEkuJyk7XG4gICAgICB9XG4gICAgICB0aGlzLmNoZWNrU2hhcGUobGF5ZXIpO1xuICAgICAgdGhpcy5vdXRwdXRzID0gW291dHB1dFRlbnNvciBhcyBTeW1ib2xpY1RlbnNvcl07XG4gICAgICAvLyB1cGRhdGUgc2VsZi5pbmJvdW5kX25vZGVzXG4gICAgICB0aGlzLmluYm91bmROb2Rlc1swXS5vdXRwdXRUZW5zb3JzID0gdGhpcy5vdXRwdXRzO1xuICAgICAgdGhpcy5pbmJvdW5kTm9kZXNbMF0ub3V0cHV0U2hhcGVzID0gW3RoaXMub3V0cHV0c1swXS5zaGFwZV07XG4gICAgfVxuXG4gICAgdGhpcy5sYXllcnMucHVzaChsYXllcik7XG4gICAgdGhpcy5idWlsdCA9IGZhbHNlO1xuICB9XG5cbiAgLyoqXG4gICAqIFJlbW92ZXMgdGhlIGxhc3QgbGF5ZXIgaW4gdGhlIG1vZGVsLlxuICAgKlxuICAgKiBAZXhjZXB0aW9uIFR5cGVFcnJvciBpZiB0aGVyZSBhcmUgbm8gbGF5ZXJzIGluIHRoZSBtb2RlbC5cbiAgICovXG4gIHBvcCgpOiB2b2lkIHtcbiAgICBpZiAodGhpcy5sYXllcnMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKCdUaGVyZSBhcmUgbm8gbGF5ZXJzIGluIHRoZSBtb2RlbC4nKTtcbiAgICB9XG5cbiAgICB0aGlzLmxheWVycy5wb3AoKTtcbiAgICBpZiAodGhpcy5sYXllcnMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aGlzLm91dHB1dHMgPSBbXTtcbiAgICAgIHRoaXMuaW5ib3VuZE5vZGVzID0gW107XG4gICAgICB0aGlzLm91dGJvdW5kTm9kZXMgPSBbXTtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgbGFzdExheWVySW5kZXggPSB0aGlzLmxheWVycy5sZW5ndGggLSAxO1xuICAgICAgdGhpcy5sYXllcnNbbGFzdExheWVySW5kZXhdLm91dGJvdW5kTm9kZXMgPSBbXTtcbiAgICAgIHRoaXMub3V0cHV0cyA9IFt0aGlzLmxheWVyc1tsYXN0TGF5ZXJJbmRleF0ub3V0cHV0IGFzIFN5bWJvbGljVGVuc29yXTtcbiAgICAgIC8vIHVwZGF0ZSBzZWxmLmluYm91bmRfbm9kZXNcbiAgICAgIHRoaXMuaW5ib3VuZE5vZGVzWzBdLm91dHB1dFRlbnNvcnMgPSB0aGlzLm91dHB1dHM7XG4gICAgICB0aGlzLmluYm91bmROb2Rlc1swXS5vdXRwdXRTaGFwZXMgPSBbdGhpcy5vdXRwdXRzWzBdLnNoYXBlXTtcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgaWYgKHRoaXMubW9kZWwgPT0gbnVsbCkge1xuICAgICAgdGhpcy5idWlsZCgpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5tb2RlbC5jYWxsKGlucHV0cywga3dhcmdzKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGJ1aWxkKGlucHV0U2hhcGU/OiBTaGFwZXxTaGFwZVtdKSB7XG4gICAgLy8gQ2FsbCBgZ2V0RXhhY3RseU9uZVNoYXBlYCB3aXRob3V0IHVzaW5nIGl0cyByZXR1cm4gdmFsdWUsXG4gICAgLy8gdG8gdmVyaWZ5IHRoYXQgZXhhY3RseSBvbmUgaW5wdXQgc2hhcGUgaXMgcHJvdmlkZWQuXG4gICAgZ2V0RXhhY3RseU9uZVNoYXBlKGlucHV0U2hhcGUpO1xuXG4gICAgaWYgKHRoaXMuaW5wdXRzLmxlbmd0aCA9PT0gMCB8fCB0aGlzLm91dHB1dHMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKFxuICAgICAgICAgICdTZXF1ZW50aWFsIG1vZGVsIGNhbm5vdCBiZSBidWlsdDogbW9kZWwgaXMgZW1wdHkuJyArXG4gICAgICAgICAgJyBBZGQgc29tZSBsYXllcnMgZmlyc3QuJyk7XG4gICAgfVxuICAgIC8vIGFjdHVhbGx5IGNyZWF0ZSB0aGUgbW9kZWxcbiAgICB0aGlzLm1vZGVsID0gbmV3IExheWVyc01vZGVsKHtcbiAgICAgIGlucHV0czogdGhpcy5pbnB1dHMsXG4gICAgICBvdXRwdXRzOiB0aGlzLm91dHB1dHNbMF0sXG4gICAgICBuYW1lOiB0aGlzLm5hbWUgKyAnX21vZGVsJ1xuICAgIH0pO1xuICAgIHRoaXMubW9kZWwudHJhaW5hYmxlID0gdGhpcy50cmFpbmFibGU7XG5cbiAgICAvLyBtaXJyb3IgbW9kZWwgYXR0cmlidXRlc1xuICAgIHRoaXMuc3VwcG9ydHNNYXNraW5nID0gdGhpcy5tb2RlbC5zdXBwb3J0c01hc2tpbmc7XG4gICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBBZGQgY2FjaGVzXG4gICAgdGhpcy5pbnB1dExheWVycyA9IHRoaXMubW9kZWwuaW5wdXRMYXllcnM7XG4gICAgdGhpcy5pbnB1dExheWVyc05vZGVJbmRpY2VzID0gdGhpcy5tb2RlbC5pbnB1dExheWVyc05vZGVJbmRpY2VzO1xuICAgIHRoaXMuaW5wdXRMYXllcnNUZW5zb3JJbmRpY2VzID0gdGhpcy5tb2RlbC5pbnB1dExheWVyc1RlbnNvckluZGljZXM7XG4gICAgdGhpcy5vdXRwdXRMYXllcnMgPSB0aGlzLm1vZGVsLm91dHB1dExheWVycztcbiAgICB0aGlzLm91dHB1dExheWVyc05vZGVJbmRpY2VzID0gdGhpcy5tb2RlbC5vdXRwdXRMYXllcnNOb2RlSW5kaWNlcztcbiAgICB0aGlzLm91dHB1dExheWVyc1RlbnNvckluZGljZXMgPSB0aGlzLm1vZGVsLm91dHB1dExheWVyc1RlbnNvckluZGljZXM7XG4gICAgdGhpcy5ub2Rlc0J5RGVwdGggPSB0aGlzLm1vZGVsLm5vZGVzQnlEZXB0aDtcbiAgICB0aGlzLmNvbnRhaW5lck5vZGVzID0gdGhpcy5tb2RlbC5jb250YWluZXJOb2RlcztcbiAgICB0aGlzLm91dHB1dE5hbWVzID0gdGhpcy5tb2RlbC5vdXRwdXROYW1lcztcbiAgICB0aGlzLmlucHV0TmFtZXMgPSB0aGlzLm1vZGVsLmlucHV0TmFtZXM7XG4gICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBBZGQgZmVlZElucHV0TmFtZXMsIGZlZWRJbnB1dHMsIGlmIG5lZWRlZC5cbiAgICAvLyBUT0RPKG1pY2hhZWx0ZXJyeSk6IEFkZCBjYWxsYmFja01vZGVsIGlmIG5lZWRlZC5cbiAgICB0aGlzLmJ1aWx0ID0gdHJ1ZTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNvdW50UGFyYW1zKCk6IG51bWJlciB7XG4gICAgaWYgKCF0aGlzLmJ1aWx0KSB7XG4gICAgICB0aGlzLmJ1aWxkKCk7XG4gICAgfVxuICAgIHJldHVybiBzdXBlci5jb3VudFBhcmFtcygpO1xuICB9XG5cbiAgLyoqXG4gICAqIFByaW50IGEgdGV4dCBzdW1tYXJ5IG9mIHRoZSBTZXF1ZW50aWFsIG1vZGVsJ3MgbGF5ZXJzLlxuICAgKlxuICAgKiBUaGUgc3VtbWFyeSBpbmNsdWRlc1xuICAgKiAtIE5hbWUgYW5kIHR5cGUgb2YgYWxsIGxheWVycyB0aGF0IGNvbXByaXNlIHRoZSBtb2RlbC5cbiAgICogLSBPdXRwdXQgc2hhcGUocykgb2YgdGhlIGxheWVyc1xuICAgKiAtIE51bWJlciBvZiB3ZWlnaHQgcGFyYW1ldGVycyBvZiBlYWNoIGxheWVyXG4gICAqIC0gVGhlIHRvdGFsIG51bWJlciBvZiB0cmFpbmFibGUgYW5kIG5vbi10cmFpbmFibGUgcGFyYW1ldGVycyBvZiB0aGVcbiAgICogbW9kZWwuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCgpO1xuICAgKiBtb2RlbC5hZGQoXG4gICAqICAgICB0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxMDAsIGlucHV0U2hhcGU6IFsxMF0sIGFjdGl2YXRpb246ICdyZWx1J30pKTtcbiAgICogbW9kZWwuYWRkKHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGFjdGl2YXRpb246ICdzaWdtb2lkJ30pKTtcbiAgICpcbiAgICogbW9kZWwuc3VtbWFyeSgpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIGxpbmVMZW5ndGggQ3VzdG9tIGxpbmUgbGVuZ3RoLCBpbiBudW1iZXIgb2YgY2hhcmFjdGVycy5cbiAgICogQHBhcmFtIHBvc2l0aW9ucyBDdXN0b20gd2lkdGhzIG9mIGVhY2ggb2YgdGhlIGNvbHVtbnMsIGFzIGVpdGhlclxuICAgKiAgIGZyYWN0aW9ucyBvZiBgbGluZUxlbmd0aGAgKGUuZy4sIGBbMC41LCAwLjc1LCAxXWApIG9yIGFic29sdXRlIG51bWJlclxuICAgKiAgIG9mIGNoYXJhY3RlcnMgKGUuZy4sIGBbMzAsIDUwLCA2NV1gKS4gRWFjaCBudW1iZXIgY29ycmVzcG9uZHMgdG9cbiAgICogICByaWdodC1tb3N0IChpLmUuLCBlbmRpbmcpIHBvc2l0aW9uIG9mIGEgY29sdW1uLlxuICAgKiBAcGFyYW0gcHJpbnRGbiBDdXN0b20gcHJpbnQgZnVuY3Rpb24uIENhbiBiZSB1c2VkIHRvIHJlcGxhY2UgdGhlIGRlZmF1bHRcbiAgICogICBgY29uc29sZS5sb2dgLiBGb3IgZXhhbXBsZSwgeW91IGNhbiB1c2UgYHggPT4ge31gIHRvIG11dGUgdGhlIHByaW50ZWRcbiAgICogICBtZXNzYWdlcyBpbiB0aGUgY29uc29sZS5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIG92ZXJyaWRlIHN1bW1hcnkoXG4gICAgICBsaW5lTGVuZ3RoPzogbnVtYmVyLCBwb3NpdGlvbnM/OiBudW1iZXJbXSxcbiAgICAgIHByaW50Rm46XG4gICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgKG1lc3NhZ2U/OiBhbnksIC4uLm9wdGlvbmFsUGFyYW1zOiBhbnlbXSkgPT4gdm9pZCA9IGNvbnNvbGUubG9nKSB7XG4gICAgaWYgKCF0aGlzLmJ1aWx0KSB7XG4gICAgICB0aGlzLmJ1aWxkKCk7XG4gICAgfVxuICAgIHN1cGVyLnN1bW1hcnkobGluZUxlbmd0aCwgcG9zaXRpb25zLCBwcmludEZuKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTZXRzIHRoZSB3ZWlnaHRzIG9mIHRoZSBtb2RlbC5cbiAgICpcbiAgICogQHBhcmFtIHdlaWdodHMgU2hvdWxkIGJlIGEgbGlzdCBvZiBUZW5zb3JzIHdpdGggc2hhcGVzIGFuZCB0eXBlcyBtYXRjaGluZ1xuICAgKiAgIHRoZSBvdXRwdXQgb2YgYG1vZGVsLmdldFdlaWdodHMoKWAuXG4gICAqL1xuICBvdmVycmlkZSBzZXRXZWlnaHRzKHdlaWdodHM6IFRlbnNvcltdKTogdm9pZCB7XG4gICAgaWYgKHRoaXMubW9kZWwgPT0gbnVsbCkge1xuICAgICAgdGhpcy5idWlsZCgpO1xuICAgIH1cbiAgICB0aGlzLm1vZGVsLnNldFdlaWdodHMod2VpZ2h0cyk7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyB0aGUgbG9zcyB2YWx1ZSAmIG1ldHJpY3MgdmFsdWVzIGZvciB0aGUgbW9kZWwgaW4gdGVzdCBtb2RlLlxuICAgKlxuICAgKiBMb3NzIGFuZCBtZXRyaWNzIGFyZSBzcGVjaWZpZWQgZHVyaW5nIGBjb21waWxlKClgLCB3aGljaCBuZWVkcyB0byBoYXBwZW5cbiAgICogYmVmb3JlIGNhbGxzIHRvIGBldmFsdWF0ZSgpYC5cbiAgICpcbiAgICogQ29tcHV0YXRpb24gaXMgZG9uZSBpbiBiYXRjaGVzLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoe1xuICAgKiAgIGxheWVyczogW3RmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFsxMF19KV1cbiAgICogfSk7XG4gICAqIG1vZGVsLmNvbXBpbGUoe29wdGltaXplcjogJ3NnZCcsIGxvc3M6ICdtZWFuU3F1YXJlZEVycm9yJ30pO1xuICAgKiBjb25zdCByZXN1bHQgPSBtb2RlbC5ldmFsdWF0ZSh0Zi5vbmVzKFs4LCAxMF0pLCB0Zi5vbmVzKFs4LCAxXSksIHtcbiAgICogICBiYXRjaFNpemU6IDQsXG4gICAqIH0pO1xuICAgKiByZXN1bHQucHJpbnQoKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSB4IGB0Zi5UZW5zb3JgIG9mIHRlc3QgZGF0YSwgb3IgYW4gYEFycmF5YCBvZiBgdGYuVGVuc29yYHMgaWYgdGhlXG4gICAqIG1vZGVsIGhhcyBtdWx0aXBsZSBpbnB1dHMuXG4gICAqIEBwYXJhbSB5IGB0Zi5UZW5zb3JgIG9mIHRhcmdldCBkYXRhLCBvciBhbiBgQXJyYXlgIG9mIGB0Zi5UZW5zb3JgcyBpZiB0aGVcbiAgICogbW9kZWwgaGFzIG11bHRpcGxlIG91dHB1dHMuXG4gICAqIEBwYXJhbSBhcmdzIEEgYE1vZGVsRXZhbHVhdGVDb25maWdgLCBjb250YWluaW5nIG9wdGlvbmFsIGZpZWxkcy5cbiAgICpcbiAgICogQHJldHVybiBgU2NhbGFyYCB0ZXN0IGxvc3MgKGlmIHRoZSBtb2RlbCBoYXMgYSBzaW5nbGUgb3V0cHV0IGFuZCBub1xuICAgKiAgIG1ldHJpY3MpIG9yIGBBcnJheWAgb2YgYFNjYWxhcmBzIChpZiB0aGUgbW9kZWwgaGFzIG11bHRpcGxlIG91dHB1dHNcbiAgICogICBhbmQvb3IgbWV0cmljcykuIFRoZSBhdHRyaWJ1dGUgYG1vZGVsLm1ldHJpY3NOYW1lc2BcbiAgICogICB3aWxsIGdpdmUgeW91IHRoZSBkaXNwbGF5IGxhYmVscyBmb3IgdGhlIHNjYWxhciBvdXRwdXRzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgb3ZlcnJpZGUgZXZhbHVhdGUoXG4gICAgICB4OiBUZW5zb3J8VGVuc29yW10sIHk6IFRlbnNvcnxUZW5zb3JbXSxcbiAgICAgIGFyZ3M6IE1vZGVsRXZhbHVhdGVBcmdzID0ge30pOiBTY2FsYXJ8U2NhbGFyW10ge1xuICAgIGlmICghdGhpcy5idWlsdCkge1xuICAgICAgdGhyb3cgbmV3IFJ1bnRpbWVFcnJvcihcbiAgICAgICAgICAnVGhlIG1vZGVsIG5lZWRzIHRvIGJlIGNvbXBpbGVkIGJlZm9yZSBiZWluZyB1c2VkLicpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5tb2RlbC5ldmFsdWF0ZSh4LCB5LCBhcmdzKTtcbiAgfVxuXG4gIC8vIFRPRE8oY2Fpcyk6IEFkZCBjb2RlIHNuaXBwZXQgYmVsb3cgb25jZSByZWFsIGRhdGFzZXQgb2JqZWN0cyBhcmVcbiAgLy8gICBhdmFpbGFibGUuXG4gIC8qKlxuICAgKiBFdmFsdWF0ZSBtb2RlbCB1c2luZyBhIGRhdGFzZXQgb2JqZWN0LlxuICAgKlxuICAgKiBOb3RlOiBVbmxpa2UgYGV2YWx1YXRlKClgLCB0aGlzIG1ldGhvZCBpcyBhc3luY2hyb25vdXMgKGBhc3luY2ApLlxuICAgKlxuICAgKiBAcGFyYW0gZGF0YXNldCBBIGRhdGFzZXQgb2JqZWN0LiBJdHMgYGl0ZXJhdG9yKClgIG1ldGhvZCBpcyBleHBlY3RlZFxuICAgKiAgIHRvIGdlbmVyYXRlIGEgZGF0YXNldCBpdGVyYXRvciBvYmplY3QsIHRoZSBgbmV4dCgpYCBtZXRob2Qgb2Ygd2hpY2hcbiAgICogICBpcyBleHBlY3RlZCB0byBwcm9kdWNlIGRhdGEgYmF0Y2hlcyBmb3IgZXZhbHVhdGlvbi4gVGhlIHJldHVybiB2YWx1ZVxuICAgKiAgIG9mIHRoZSBgbmV4dCgpYCBjYWxsIG91Z2h0IHRvIGNvbnRhaW4gYSBib29sZWFuIGBkb25lYCBmaWVsZCBhbmQgYVxuICAgKiAgIGB2YWx1ZWAgZmllbGQuIFRoZSBgdmFsdWVgIGZpZWxkIGlzIGV4cGVjdGVkIHRvIGJlIGFuIGFycmF5IG9mIHR3b1xuICAgKiAgIGB0Zi5UZW5zb3JgcyBvciBhbiBhcnJheSBvZiB0d28gbmVzdGVkIGB0Zi5UZW5zb3JgIHN0cnVjdHVyZXMuIFRoZSBmb3JtZXJcbiAgICogICBjYXNlIGlzIGZvciBtb2RlbHMgd2l0aCBleGFjdGx5IG9uZSBpbnB1dCBhbmQgb25lIG91dHB1dCAoZS5nLlxuICAgKiAgIGEgc2VxdWVudGlhbCBtb2RlbCkuIFRoZSBsYXR0ZXIgY2FzZSBpcyBmb3IgbW9kZWxzIHdpdGggbXVsdGlwbGVcbiAgICogICBpbnB1dHMgYW5kL29yIG11bHRpcGxlIG91dHB1dHMuIE9mIHRoZSB0d28gaXRlbXMgaW4gdGhlIGFycmF5LCB0aGVcbiAgICogICBmaXJzdCBpcyB0aGUgaW5wdXQgZmVhdHVyZShzKSBhbmQgdGhlIHNlY29uZCBpcyB0aGUgb3V0cHV0IHRhcmdldChzKS5cbiAgICogQHBhcmFtIGFyZ3MgQSBjb25maWd1cmF0aW9uIG9iamVjdCBmb3IgdGhlIGRhdGFzZXQtYmFzZWQgZXZhbHVhdGlvbi5cbiAgICogQHJldHVybnMgTG9zcyBhbmQgbWV0cmljIHZhbHVlcyBhcyBhbiBBcnJheSBvZiBgU2NhbGFyYCBvYmplY3RzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgb3ZlcnJpZGUgYXN5bmMgZXZhbHVhdGVEYXRhc2V0KGRhdGFzZXQ6IERhdGFzZXQ8e30+LFxuICAgICAgYXJnczogTW9kZWxFdmFsdWF0ZURhdGFzZXRBcmdzKTogUHJvbWlzZTxTY2FsYXJ8U2NhbGFyW10+IHtcbiAgICBpZiAoIXRoaXMuYnVpbHQpIHtcbiAgICAgIHRocm93IG5ldyBSdW50aW1lRXJyb3IoXG4gICAgICAgICAgJ1RoZSBtb2RlbCBuZWVkcyB0byBiZSBjb21waWxlZCBiZWZvcmUgYmVpbmcgdXNlZC4nKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMubW9kZWwuZXZhbHVhdGVEYXRhc2V0KGRhdGFzZXQsIGFyZ3MpO1xuICB9XG5cbiAgLyoqXG4gICAqIEdlbmVyYXRlcyBvdXRwdXQgcHJlZGljdGlvbnMgZm9yIHRoZSBpbnB1dCBzYW1wbGVzLlxuICAgKlxuICAgKiBDb21wdXRhdGlvbiBpcyBkb25lIGluIGJhdGNoZXMuXG4gICAqXG4gICAqIE5vdGU6IHRoZSBcInN0ZXBcIiBtb2RlIG9mIHByZWRpY3QoKSBpcyBjdXJyZW50bHkgbm90IHN1cHBvcnRlZC5cbiAgICogICBUaGlzIGlzIGJlY2F1c2UgdGhlIFRlbnNvckZsb3cuanMgY29yZSBiYWNrZW5kIGlzIGltcGVyYXRpdmUgb25seS5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKHtcbiAgICogICBsYXllcnM6IFt0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxLCBpbnB1dFNoYXBlOiBbMTBdfSldXG4gICAqIH0pO1xuICAgKiBtb2RlbC5wcmVkaWN0KHRmLm9uZXMoWzIsIDEwXSkpLnByaW50KCk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgZGF0YSwgYXMgYSBUZW5zb3IsIG9yIGFuIGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzIGlmXG4gICAqICAgdGhlIG1vZGVsIGhhcyBtdWx0aXBsZSBpbnB1dHMuXG4gICAqIEBwYXJhbSBjb25pZmcgQSBgTW9kZWxQcmVkaWN0Q29uZmlnYCBvYmplY3QgY29udGFpbmluZyBvcHRpb25hbCBmaWVsZHMuXG4gICAqXG4gICAqIEByZXR1cm4gYHRmLlRlbnNvcmAocykgb2YgcHJlZGljdGlvbnMuXG4gICAqXG4gICAqIEBleGNlcHRpb24gVmFsdWVFcnJvciBJbiBjYXNlIG9mIG1pc21hdGNoIGJldHdlZW4gdGhlIHByb3ZpZGVkIGlucHV0IGRhdGFcbiAgICogICBhbmQgdGhlIG1vZGVsJ3MgZXhwZWN0YXRpb25zLCBvciBpbiBjYXNlIGEgc3RhdGVmdWwgbW9kZWwgcmVjZWl2ZXMgYVxuICAgKiAgIG51bWJlciBvZiBzYW1wbGVzIHRoYXQgaXMgbm90IGEgbXVsdGlwbGUgb2YgdGhlIGJhdGNoIHNpemUuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBvdmVycmlkZSBwcmVkaWN0KHg6IFRlbnNvcnxUZW5zb3JbXSwgYXJnczogTW9kZWxQcmVkaWN0QXJncyA9IHt9KTpcbiAgICAgIFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgaWYgKHRoaXMubW9kZWwgPT0gbnVsbCkge1xuICAgICAgdGhpcy5idWlsZCgpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5tb2RlbC5wcmVkaWN0KHgsIGFyZ3MpO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgcHJlZGljdGlvbnMgZm9yIGEgc2luZ2xlIGJhdGNoIG9mIHNhbXBsZXMuXG4gICAqXG4gICAqIEBwYXJhbSB4OiBJbnB1dCBzYW1wbGVzLCBhcyBhIFRlbnNvciwgb3IgbGlzdCBvZiBUZW5zb3JzIChpZiB0aGUgbW9kZWxcbiAgICogICBoYXMgbXVsdGlwbGUgaW5wdXRzKS5cbiAgICogQHJldHVybiBUZW5zb3Iocykgb2YgcHJlZGljdGlvbnNcbiAgICovXG4gIG92ZXJyaWRlIHByZWRpY3RPbkJhdGNoKHg6IFRlbnNvcik6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgaWYgKHRoaXMubW9kZWwgPT0gbnVsbCkge1xuICAgICAgdGhpcy5idWlsZCgpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5tb2RlbC5wcmVkaWN0T25CYXRjaCh4KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTZWUgYExheWVyc01vZGVsLmNvbXBpbGVgLlxuICAgKlxuICAgKiBAcGFyYW0gYXJnc1xuICAgKi9cbiAgb3ZlcnJpZGUgY29tcGlsZShhcmdzOiBNb2RlbENvbXBpbGVBcmdzKTogdm9pZCB7XG4gICAgdGhpcy5idWlsZCgpO1xuICAgIHRoaXMubW9kZWwuY29tcGlsZShhcmdzKTtcbiAgICB0aGlzLm9wdGltaXplcl8gPSB0aGlzLm1vZGVsLm9wdGltaXplcjtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgdGhpcy5pc09wdGltaXplck93bmVkID0gKHRoaXMubW9kZWwgYXMgYW55KS5pc09wdGltaXplck93bmVkO1xuICAgIHRoaXMubG9zcyA9IHRoaXMubW9kZWwubG9zcztcbiAgICB0aGlzLm1ldHJpY3MgPSB0aGlzLm1vZGVsLm1ldHJpY3M7XG4gICAgLy8gVE9ETyhjYWlzKTogQWRkIHRoaXMubG9zc1dlaWdodHMsIHRoaXMuc2FtcGxlV2VpZ2h0TW9kZSxcbiAgICAvLyAgIHRoaXMud2VpZ2h0ZWRNZXRyaWNzLCB0aGlzLnRhcmdldHMuXG4gICAgdGhpcy5tZXRyaWNzVGVuc29ycyA9IHRoaXMubW9kZWwubWV0cmljc1RlbnNvcnM7XG4gICAgdGhpcy5tZXRyaWNzTmFtZXMgPSB0aGlzLm1vZGVsLm1ldHJpY3NOYW1lcztcbiAgICAvLyBUT0RPKGNhaXMpOiBBZGQgc2FtcGxlV2VpZ2h0cy5cbiAgfVxuXG4gIG92ZXJyaWRlIGdldCBvcHRpbWl6ZXIoKTogT3B0aW1pemVyIHtcbiAgICByZXR1cm4gdGhpcy5tb2RlbCA9PSBudWxsID8gdW5kZWZpbmVkIDogdGhpcy5tb2RlbC5vcHRpbWl6ZXI7XG4gIH1cblxuICBvdmVycmlkZSBzZXQgb3B0aW1pemVyKG9wdGltaXplcjogT3B0aW1pemVyKSB7XG4gICAgdGhpcy5tb2RlbC5vcHRpbWl6ZXIgPSBvcHRpbWl6ZXI7XG4gIH1cblxuICAvKipcbiAgICogVHJhaW5zIHRoZSBtb2RlbCBmb3IgYSBmaXhlZCBudW1iZXIgb2YgZXBvY2hzIChpdGVyYXRpb25zIG9uIGEgZGF0YXNldCkuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCh7XG4gICAqICAgbGF5ZXJzOiBbdGYubGF5ZXJzLmRlbnNlKHt1bml0czogMSwgaW5wdXRTaGFwZTogWzEwXX0pXVxuICAgKiB9KTtcbiAgICogbW9kZWwuY29tcGlsZSh7b3B0aW1pemVyOiAnc2dkJywgbG9zczogJ21lYW5TcXVhcmVkRXJyb3InfSk7XG4gICAqIGNvbnN0IGhpc3RvcnkgPSBhd2FpdCBtb2RlbC5maXQodGYub25lcyhbOCwgMTBdKSwgdGYub25lcyhbOCwgMV0pLCB7XG4gICAqICAgYmF0Y2hTaXplOiA0LFxuICAgKiAgIGVwb2NoczogM1xuICAgKiB9KTtcbiAgICogY29uc29sZS5sb2coaGlzdG9yeS5oaXN0b3J5Lmxvc3NbMF0pO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIHggYHRmLlRlbnNvcmAgb2YgdHJhaW5pbmcgZGF0YSwgb3IgYW4gYXJyYXkgb2YgYHRmLlRlbnNvcmBzIGlmIHRoZVxuICAgKiBtb2RlbCBoYXMgbXVsdGlwbGUgaW5wdXRzLiBJZiBhbGwgaW5wdXRzIGluIHRoZSBtb2RlbCBhcmUgbmFtZWQsIHlvdSBjYW5cbiAgICogYWxzbyBwYXNzIGEgZGljdGlvbmFyeSBtYXBwaW5nIGlucHV0IG5hbWVzIHRvIGB0Zi5UZW5zb3Jgcy5cbiAgICogQHBhcmFtIHkgYHRmLlRlbnNvcmAgb2YgdGFyZ2V0IChsYWJlbCkgZGF0YSwgb3IgYW4gYXJyYXkgb2YgYHRmLlRlbnNvcmBzIGlmXG4gICAqIHRoZSBtb2RlbCBoYXMgbXVsdGlwbGUgb3V0cHV0cy4gSWYgYWxsIG91dHB1dHMgaW4gdGhlIG1vZGVsIGFyZSBuYW1lZCwgeW91XG4gICAqICBjYW4gYWxzbyBwYXNzIGEgZGljdGlvbmFyeSBtYXBwaW5nIG91dHB1dCBuYW1lcyB0byBgdGYuVGVuc29yYHMuXG4gICAqIEBwYXJhbSBhcmdzICBBIGBNb2RlbEZpdENvbmZpZ2AsIGNvbnRhaW5pbmcgb3B0aW9uYWwgZmllbGRzLlxuICAgKlxuICAgKiBAcmV0dXJuIEEgYEhpc3RvcnlgIGluc3RhbmNlLiBJdHMgYGhpc3RvcnlgIGF0dHJpYnV0ZSBjb250YWlucyBhbGxcbiAgICogICBpbmZvcm1hdGlvbiBjb2xsZWN0ZWQgZHVyaW5nIHRyYWluaW5nLlxuICAgKlxuICAgKiBAZXhjZXB0aW9uIFZhbHVlRXJyb3IgSW4gY2FzZSBvZiBtaXNtYXRjaCBiZXR3ZWVuIHRoZSBwcm92aWRlZCBpbnB1dCBkYXRhXG4gICAqICAgYW5kIHdoYXQgdGhlIG1vZGVsIGV4cGVjdHMuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBvdmVycmlkZSBhc3luYyBmaXQoXG4gICAgICB4OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgICB5OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgICBhcmdzOiBNb2RlbEZpdEFyZ3MgPSB7fSk6IFByb21pc2U8SGlzdG9yeT4ge1xuICAgIGlmICghdGhpcy5idWlsdCkge1xuICAgICAgdGhyb3cgbmV3IFJ1bnRpbWVFcnJvcihcbiAgICAgICAgICAnVGhlIG1vZGVsIG5lZWRzIHRvIGJlIGNvbXBpbGVkIGJlZm9yZSAnICtcbiAgICAgICAgICAnYmVpbmcgdXNlZC4nKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMubW9kZWwuZml0KHgsIHksIGFyZ3MpO1xuICB9XG5cbiAgLyoqXG4gICAqIFRyYWlucyB0aGUgbW9kZWwgdXNpbmcgYSBkYXRhc2V0IG9iamVjdC5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgeEFycmF5ID0gW1xuICAgKiAgIFsxLCAxLCAxLCAxLCAxLCAxLCAxLCAxLCAxXSxcbiAgICogICBbMSwgMSwgMSwgMSwgMSwgMSwgMSwgMSwgMV0sXG4gICAqICAgWzEsIDEsIDEsIDEsIDEsIDEsIDEsIDEsIDFdLFxuICAgKiAgIFsxLCAxLCAxLCAxLCAxLCAxLCAxLCAxLCAxXSxcbiAgICogXTtcbiAgICogY29uc3QgeUFycmF5ID0gWzEsIDEsIDEsIDFdO1xuICAgKiAvLyBDcmVhdGUgYSBkYXRhc2V0IGZyb20gdGhlIEphdmFTY3JpcHQgYXJyYXkuXG4gICAqIGNvbnN0IHhEYXRhc2V0ID0gdGYuZGF0YS5hcnJheSh4QXJyYXkpO1xuICAgKiBjb25zdCB5RGF0YXNldCA9IHRmLmRhdGEuYXJyYXkoeUFycmF5KTtcbiAgICogLy8gWmlwIGNvbWJpbmVzIHRoZSBgeGAgYW5kIGB5YCBEYXRhc2V0cyBpbnRvIGEgc2luZ2xlIERhdGFzZXQsIHRoZVxuICAgKiAvLyBpdGVyYXRvciBvZiB3aGljaCB3aWxsIHJldHVybiBhbiBvYmplY3QgY29udGFpbmluZyBvZiB0d28gdGVuc29ycyxcbiAgICogLy8gY29ycmVzcG9uZGluZyB0byBgeGAgYW5kIGB5YC4gIFRoZSBjYWxsIHRvIGBiYXRjaCg0KWAgd2lsbCBidW5kbGVcbiAgICogLy8gZm91ciBzdWNoIHNhbXBsZXMgaW50byBhIHNpbmdsZSBvYmplY3QsIHdpdGggdGhlIHNhbWUga2V5cyBub3cgcG9pbnRpbmdcbiAgICogLy8gdG8gdGVuc29ycyB0aGF0IGhvbGQgNCBleGFtcGxlcywgb3JnYW5pemVkIGFsb25nIHRoZSBiYXRjaCBkaW1lbnNpb24uXG4gICAqIC8vIFRoZSBjYWxsIHRvIGBzaHVmZmxlKDQpYCBjYXVzZXMgZWFjaCBpdGVyYXRpb24gdGhyb3VnaCB0aGUgZGF0YXNldCB0b1xuICAgKiAvLyBoYXBwZW4gaW4gYSBkaWZmZXJlbnQgb3JkZXIuICBUaGUgc2l6ZSBvZiB0aGUgc2h1ZmZsZSB3aW5kb3cgaXMgNC5cbiAgICogY29uc3QgeHlEYXRhc2V0ID0gdGYuZGF0YS56aXAoe3hzOiB4RGF0YXNldCwgeXM6IHlEYXRhc2V0fSlcbiAgICogICAgIC5iYXRjaCg0KVxuICAgKiAgICAgLnNodWZmbGUoNCk7XG4gICAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCh7XG4gICAqICAgbGF5ZXJzOiBbdGYubGF5ZXJzLmRlbnNlKHt1bml0czogMSwgaW5wdXRTaGFwZTogWzldfSldXG4gICAqIH0pO1xuICAgKiBtb2RlbC5jb21waWxlKHtvcHRpbWl6ZXI6ICdzZ2QnLCBsb3NzOiAnbWVhblNxdWFyZWRFcnJvcid9KTtcbiAgICogY29uc3QgaGlzdG9yeSA9IGF3YWl0IG1vZGVsLmZpdERhdGFzZXQoeHlEYXRhc2V0LCB7XG4gICAqICAgZXBvY2hzOiA0LFxuICAgKiAgIGNhbGxiYWNrczoge29uRXBvY2hFbmQ6IChlcG9jaCwgbG9ncykgPT4gY29uc29sZS5sb2cobG9ncy5sb3NzKX1cbiAgICogfSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0gZGF0YXNldCBBIGRhdGFzZXQgb2JqZWN0LiBJdHMgYGl0ZXJhdG9yKClgIG1ldGhvZCBpcyBleHBlY3RlZCB0b1xuICAgKiAgIGdlbmVyYXRlIGEgZGF0YXNldCBpdGVyYXRvciBvYmplY3QsIHRoZSBgbmV4dCgpYCBtZXRob2Qgb2Ygd2hpY2ggaXNcbiAgICogICBleHBlY3RlZCB0byBwcm9kdWNlIGRhdGEgYmF0Y2hlcyBmb3IgZXZhbHVhdGlvbi4gVGhlIHJldHVybiB2YWx1ZSBvZiB0aGVcbiAgICogICBgbmV4dCgpYCBjYWxsIG91Z2h0IHRvIGNvbnRhaW4gYSBib29sZWFuIGBkb25lYCBmaWVsZCBhbmQgYSBgdmFsdWVgXG4gICAqICAgZmllbGQuXG4gICAqXG4gICAqICAgVGhlIGB2YWx1ZWAgZmllbGQgaXMgZXhwZWN0ZWQgdG8gYmUgYW4gb2JqZWN0IG9mIHdpdGggZmllbGRzXG4gICAqICAgYHhzYCBhbmQgYHlzYCwgd2hpY2ggcG9pbnQgdG8gdGhlIGZlYXR1cmUgdGVuc29yIGFuZCB0aGUgdGFyZ2V0IHRlbnNvcixcbiAgICogICByZXNwZWN0aXZlbHkuIFRoaXMgY2FzZSBpcyBmb3IgbW9kZWxzIHdpdGggZXhhY3RseSBvbmUgaW5wdXQgYW5kIG9uZVxuICAgKiAgIG91dHB1dCAoZS5nLiBhIHNlcXVlbnRpYWwgbW9kZWwpLiBGb3IgZXhhbXBsZTpcbiAgICogICBgYGBqc1xuICAgKiAgIHt2YWx1ZToge3hzOiB4c1RlbnNvciwgeXM6IHlzVGVuc29yfSwgZG9uZTogZmFsc2V9XG4gICAqICAgYGBgXG4gICAqXG4gICAqICAgSWYgdGhlIG1vZGVsIGhhcyBtdWx0aXBsZSBpbnB1dHMsIHRoZSBgeHNgIGZpZWxkIG9mIGB2YWx1ZWAgc2hvdWxkXG4gICAqICAgYmUgYW4gb2JqZWN0IG1hcHBpbmcgaW5wdXQgbmFtZXMgdG8gdGhlaXIgcmVzcGVjdGl2ZSBmZWF0dXJlIHRlbnNvcnMuXG4gICAqICAgRm9yIGV4YW1wbGU6XG4gICAqICAgYGBganNcbiAgICogICB7XG4gICAqICAgICB2YWx1ZToge1xuICAgKiAgICAgICB4czoge1xuICAgKiAgICAgICAgIGlucHV0XzE6IHhzVGVuc29yMSxcbiAgICogICAgICAgICBpbnB1dF8yOiB4c1RlbnNvcjJcbiAgICogICAgICAgfSxcbiAgICogICAgICAgeXM6IHlzVGVuc29yXG4gICAqICAgICB9LFxuICAgKiAgICAgZG9uZTogZmFsc2VcbiAgICogICB9XG4gICAqICAgYGBgXG4gICAqICAgSWYgdGhlIG1vZGVsIGhhcyBtdWx0aXBsZSBvdXRwdXRzLCB0aGUgYHlzYCBmaWVsZCBvZiBgdmFsdWVgIHNob3VsZFxuICAgKiAgIGJlIGFuIG9iamVjdCBtYXBwaW5nIG91dHB1dCBuYW1lcyB0byB0aGVpciByZXNwZWN0aXZlIHRhcmdldCB0ZW5zb3JzLlxuICAgKiAgIEZvciBleGFtcGxlOlxuICAgKiAgIGBgYGpzXG4gICAqICAge1xuICAgKiAgICAgdmFsdWU6IHtcbiAgICogICAgICAgeHM6IHhzVGVuc29yLFxuICAgKiAgICAgICB5czoge1xuICAgKiAgICAgICAgIG91dHB1dF8xOiB5c1RlbnNvcjEsXG4gICAqICAgICAgICAgb3V0cHV0XzI6IHlzVGVuc29yMlxuICAgKiAgICAgICB9LFxuICAgKiAgICAgfSxcbiAgICogICAgIGRvbmU6IGZhbHNlXG4gICAqICAgfVxuICAgKiAgIGBgYFxuICAgKiBAcGFyYW0gYXJncyBBIGBNb2RlbEZpdERhdGFzZXRBcmdzYCwgY29udGFpbmluZyBvcHRpb25hbCBmaWVsZHMuXG4gICAqXG4gICAqIEByZXR1cm4gQSBgSGlzdG9yeWAgaW5zdGFuY2UuIEl0cyBgaGlzdG9yeWAgYXR0cmlidXRlIGNvbnRhaW5zIGFsbFxuICAgKiAgIGluZm9ybWF0aW9uIGNvbGxlY3RlZCBkdXJpbmcgdHJhaW5pbmcuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3NlcycsIGlnbm9yZUNJOiB0cnVlfVxuICAgKi9cbiAgb3ZlcnJpZGUgYXN5bmMgZml0RGF0YXNldDxUPihkYXRhc2V0OiBEYXRhc2V0PFQ+LFxuICAgICAgYXJnczogTW9kZWxGaXREYXRhc2V0QXJnczxUPik6IFByb21pc2U8SGlzdG9yeT4ge1xuICAgIGlmICghdGhpcy5idWlsdCkge1xuICAgICAgdGhyb3cgbmV3IFJ1bnRpbWVFcnJvcihcbiAgICAgICAgICAnVGhlIG1vZGVsIG5lZWRzIHRvIGJlIGNvbXBpbGVkIGJlZm9yZSAnICtcbiAgICAgICAgICAnYmVpbmcgdXNlZC4nKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMubW9kZWwuZml0RGF0YXNldChkYXRhc2V0LCBhcmdzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSdW5zIGEgc2luZ2xlIGdyYWRpZW50IHVwZGF0ZSBvbiBhIHNpbmdsZSBiYXRjaCBvZiBkYXRhLlxuICAgKlxuICAgKiBUaGlzIG1ldGhvZCBkaWZmZXJzIGZyb20gYGZpdCgpYCBhbmQgYGZpdERhdGFzZXQoKWAgaW4gdGhlIGZvbGxvd2luZ1xuICAgKiByZWdhcmRzOlxuICAgKiAgIC0gSXQgb3BlcmF0ZXMgb24gZXhhY3RseSBvbmUgYmF0Y2ggb2YgZGF0YS5cbiAgICogICAtIEl0IHJldHVybnMgb25seSB0aGUgbG9zcyBhbmQgbWV0cmljIHZhbHVlcywgaW5zdGVhZCBvZlxuICAgKiAgICAgcmV0dXJuaW5nIHRoZSBiYXRjaC1ieS1iYXRjaCBsb3NzIGFuZCBtZXRyaWMgdmFsdWVzLlxuICAgKiAgIC0gSXQgZG9lc24ndCBzdXBwb3J0IGZpbmUtZ3JhaW5lZCBvcHRpb25zIHN1Y2ggYXMgdmVyYm9zaXR5IGFuZFxuICAgKiAgICAgY2FsbGJhY2tzLlxuICAgKlxuICAgKiBAcGFyYW0geCBJbnB1dCBkYXRhLiBJdCBjb3VsZCBiZSBvbmUgb2YgdGhlIGZvbGxvd2luZzpcbiAgICogICAtIEEgYHRmLlRlbnNvcmAsIG9yIGFuIEFycmF5IG9mIGB0Zi5UZW5zb3JgcyAoaW4gY2FzZSB0aGUgbW9kZWwgaGFzXG4gICAqICAgICBtdWx0aXBsZSBpbnB1dHMpLlxuICAgKiAgIC0gQW4gT2JqZWN0IG1hcHBpbmcgaW5wdXQgbmFtZXMgdG8gY29ycmVzcG9uZGluZyBgdGYuVGVuc29yYCAoaWYgdGhlXG4gICAqICAgICBtb2RlbCBoYXMgbmFtZWQgaW5wdXRzKS5cbiAgICogQHBhcmFtIHkgVGFyZ2V0IGRhdGEuIEl0IGNvdWxkIGJlIGVpdGhlciBhIGB0Zi5UZW5zb3JgIG9yIG11bHRpcGxlXG4gICAqICAgYHRmLlRlbnNvcmBzLiBJdCBzaG91bGQgYmUgY29uc2lzdGVudCB3aXRoIGB4YC5cbiAgICogQHJldHVybnMgVHJhaW5pbmcgbG9zcyBvciBsb3NzZXMgKGluIGNhc2UgdGhlIG1vZGVsIGhhc1xuICAgKiAgIG11bHRpcGxlIG91dHB1dHMpLCBhbG9uZyB3aXRoIG1ldHJpY3MgKGlmIGFueSksIGFzIG51bWJlcnMuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBvdmVycmlkZSBhc3luYyB0cmFpbk9uQmF0Y2goXG4gICAgICB4OiBUZW5zb3J8VGVuc29yW118e1tpbnB1dE5hbWU6IHN0cmluZ106IFRlbnNvcn0sXG4gICAgICB5OiBUZW5zb3J8VGVuc29yW118XG4gICAgICB7W2lucHV0TmFtZTogc3RyaW5nXTogVGVuc29yfSk6IFByb21pc2U8bnVtYmVyfG51bWJlcltdPiB7XG4gICAgcmV0dXJuIHRoaXMubW9kZWwudHJhaW5PbkJhdGNoKHgsIHkpO1xuICB9XG5cbiAgLyogU2VlIHBhcmVudCBjbGFzcyBmb3IgSnNEb2MgKi9cbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBmcm9tQ29uZmlnPFQgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZT4oXG4gICAgICBjbHM6IHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlQ29uc3RydWN0b3I8VD4sXG4gICAgICBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCxcbiAgICAgIGN1c3RvbU9iamVjdHMgPSB7fSBhcyBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QsXG4gICAgICBmYXN0V2VpZ2h0SW5pdCA9IGZhbHNlKTogVCB7XG4gICAgbGV0IGNvbmZpZ0FycmF5OiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3RBcnJheTtcbiAgICBsZXQgZXh0cmFNb2RlbENvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge307XG4gICAgaWYgKGNvbmZpZyBpbnN0YW5jZW9mIEFycmF5KSB7XG4gICAgICBpZiAoIShjb25maWdbMF0uY2xhc3NOYW1lICE9IG51bGwpIHx8XG4gICAgICAgICAgY29uZmlnWzBdWydjbGFzc05hbWUnXSA9PT0gJ01lcmdlJykge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcignTGVnYWN5IHNlcmlhbGl6YXRpb24gZm9ybWF0IG5vdCBzdXBwb3J0ZWQgeWV0LicpO1xuICAgICAgfVxuICAgICAgY29uZmlnQXJyYXkgPSBjb25maWc7XG4gICAgfSBlbHNlIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGNvbmZpZ1snbGF5ZXJzJ10gIT0gbnVsbCxcbiAgICAgICAgICAoKSA9PlxuICAgICAgICAgICAgICBgV2hlbiB0aGUgY29uZmlnIGRhdGEgZm9yIGEgU2VxdWVudGlhbCBtb2RlbCBpcyBub3QgYW4gQXJyYXksIGAgK1xuICAgICAgICAgICAgICBgaXQgbXVzdCBiZSBhbiBPYmplY3QgdGhhdCBjb250YWlucyB0aGUgJ2xheWVycycgZmllbGQuYCk7XG4gICAgICBjb25maWdBcnJheSA9IGNvbmZpZ1snbGF5ZXJzJ10gYXMgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0QXJyYXk7XG4gICAgICBkZWxldGUgY29uZmlnWydsYXllcnMnXTtcbiAgICAgIGV4dHJhTW9kZWxDb25maWcgPSBjb25maWc7XG4gICAgfVxuXG4gICAgY29uc3QgbW9kZWwgPSBuZXcgY2xzKGV4dHJhTW9kZWxDb25maWcpO1xuICAgIGlmICghKG1vZGVsIGluc3RhbmNlb2YgU2VxdWVudGlhbCkpIHtcbiAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgIGBTZXF1ZW50aWFsLmZyb21Db25maWcgY2FsbGVkIG9uIG5vbi1TZXF1ZW50aWFsIGlucHV0OiAke21vZGVsfWApO1xuICAgIH1cbiAgICBmb3IgKGNvbnN0IGNvbmYgb2YgY29uZmlnQXJyYXkpIHtcbiAgICAgIGNvbnN0IGN1c3RvbU9iamVjdHM6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHVuZGVmaW5lZDtcbiAgICAgIGNvbnN0IGxheWVyID0gZGVzZXJpYWxpemUoXG4gICAgICAgICAgICAgICAgICAgICAgICBjb25mIGFzIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCwgY3VzdG9tT2JqZWN0cyxcbiAgICAgICAgICAgICAgICAgICAgICAgIGZhc3RXZWlnaHRJbml0KSBhcyBMYXllcjtcbiAgICAgIGlmIChmYXN0V2VpZ2h0SW5pdCkge1xuICAgICAgICBsYXllci5zZXRGYXN0V2VpZ2h0SW5pdER1cmluZ0J1aWxkKHRydWUpO1xuICAgICAgfVxuICAgICAgbW9kZWwuYWRkKGxheWVyKTtcbiAgICB9XG4gICAgcmV0dXJuIG1vZGVsO1xuICB9XG5cbiAgLyoqXG4gICAqIFNldHRlciB1c2VkIGZvciBmb3JjZSBzdG9wcGluZyBvZiBMYXllcnNNb2RlbC5maXQoKSAoaS5lLiwgdHJhaW5pbmcpLlxuICAgKlxuICAgKiBFeGFtcGxlOlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoKTtcbiAgICogbW9kZWwuYWRkKHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFsxMF19KSk7XG4gICAqIG1vZGVsLmNvbXBpbGUoe2xvc3M6ICdtZWFuU3F1YXJlZEVycm9yJywgb3B0aW1pemVyOiAnc2dkJ30pO1xuICAgKiBjb25zdCB4cyA9IHRmLm9uZXMoWzgsIDEwXSk7XG4gICAqIGNvbnN0IHlzID0gdGYuemVyb3MoWzgsIDFdKTtcbiAgICpcbiAgICogY29uc3QgaGlzdG9yeSA9IGF3YWl0IG1vZGVsLmZpdCh4cywgeXMsIHtcbiAgICogICBlcG9jaHM6IDEwLFxuICAgKiAgIGNhbGxiYWNrczoge1xuICAgKiAgICAgb25FcG9jaEVuZDogYXN5bmMgKGVwb2NoLCBsb2dzKSA9PiB7XG4gICAqICAgICAgIGlmIChlcG9jaCA9PT0gMikge1xuICAgKiAgICAgICAgIG1vZGVsLnN0b3BUcmFpbmluZyA9IHRydWU7XG4gICAqICAgICAgIH1cbiAgICogICAgIH1cbiAgICogICB9XG4gICAqIH0pO1xuICAgKlxuICAgKiAvLyBUaGVyZSBzaG91bGQgYmUgb25seSAzIHZhbHVlcyBpbiB0aGUgbG9zcyBhcnJheSwgaW5zdGVhZCBvZiAxMCB2YWx1ZXMsXG4gICAqIC8vIGR1ZSB0byB0aGUgc3RvcHBpbmcgYWZ0ZXIgMyBlcG9jaHMuXG4gICAqIGNvbnNvbGUubG9nKGhpc3RvcnkuaGlzdG9yeS5sb3NzKTtcbiAgICogYGBgXG4gICAqL1xuICBvdmVycmlkZSBzZXQgc3RvcFRyYWluaW5nKHN0b3A6IGJvb2xlYW4pIHtcbiAgICAvLyBUT0RPKGNhaXMpOiBXaGVuIHJlZmFjdG9yaW5nIHRvIHJlbW92ZSB0aGUgY29tcG9zaXRpb24gcGF0dGVybiBoYXBwZW5zLFxuICAgIC8vIHJlbW92ZSB0aGlzIG1ldGhvZCBvdmVycmlkaW5nLlxuICAgIGlmICh0aGlzLm1vZGVsID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdDYW5ub3Qgc2V0IHRoZSBzdG9wVHJhaW5pbmcgcHJvcGVydHkgb2YgYSBzZXF1ZW50aWFsIG1vZGVsIGJlZm9yZSAnICtcbiAgICAgICAgICAnaXQgaXMgY29tcGlsZWQuJyk7XG4gICAgfVxuICAgIHRoaXMubW9kZWwuc3RvcFRyYWluaW5nID0gc3RvcDtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldCBzdG9wVHJhaW5pbmcoKTogYm9vbGVhbiB7XG4gICAgaWYgKHRoaXMubW9kZWwgPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ0Nhbm5vdCBnZXQgdGhlIHN0b3BUcmFpbmluZyBwcm9wZXJ0eSBvZiBhIHNlcXVlbnRpYWwgbW9kZWwgYmVmb3JlICcgK1xuICAgICAgICAgICdpdCBpcyBjb21waWxlZC4nKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMubW9kZWwuc3RvcFRyYWluaW5nO1xuICB9XG5cbiAgLy8gVE9ETyhjYWlzKTogT3ZlcnJpZGUgZ2V0IHRyYWluYWJsZVdlaWdodHMoKSBoZXJlXG5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICBvdmVycmlkZSBnZXRDb25maWcoKTogYW55IHtcbiAgICAvLyBOT1RFKGNhaXMpOiBXZSBvdmVycmlkZSB0aGUgcmV0dXJuIHR5cGUgb2YgZ2V0Q29uZmlnKCkgdG8gYGFueWAgaGVyZSxcbiAgICAvLyAgIGJlY2F1c2UgdGhlIGBTZXF1ZW50aWFsYCBjbGFzcyBpcyBhIHNwZWNpYWwgY2FzZSBhbW9uZyBgQ29udGFpbmVyYFxuICAgIC8vICAgc3VidHlwZXMgaW4gdGhhdCBpdHMgZ2V0Q29uZmlnKCkgbWV0aG9kIHJldHVybnMgYW4gQXJyYXkgKG5vdCBhXG4gICAgLy8gICBkaWN0KS5cbiAgICBjb25zdCBsYXllcnM6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdFtdID0gW107XG4gICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLmxheWVycykge1xuICAgICAgY29uc3QgZGljdDogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge307XG4gICAgICBkaWN0WydjbGFzc05hbWUnXSA9IGxheWVyLmdldENsYXNzTmFtZSgpO1xuICAgICAgZGljdFsnY29uZmlnJ10gPSBsYXllci5nZXRDb25maWcoKTtcbiAgICAgIGxheWVycy5wdXNoKGRpY3QpO1xuICAgIH1cbiAgICByZXR1cm4ge25hbWU6IHRoaXMubmFtZSwgbGF5ZXJzfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFNlcXVlbnRpYWwpO1xuIl19