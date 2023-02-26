/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* Original source: keras/engine/topology.py */
import { tidy } from '@tensorflow/tfjs-core';
import { getUid } from '../backend/state';
import { NotImplementedError, RuntimeError, ValueError } from '../errors';
import { deserialize as deserializeLayer } from '../layers/serialization';
import * as generic_utils from '../utils/generic_utils';
import { convertTsToPythonic } from '../utils/serialization_utils';
import * as types_utils from '../utils/types_utils';
import { batchSetValue } from '../variables';
import { version as layersVersion } from '../version';
import { execute, FeedDict } from './executor';
import { InputLayer } from './input_layer';
import { Layer, Node } from './topology';
/**
 * A Container is a directed acyclic graph of layers.
 *
 * It is the topological form of a "model". A LayersModel
 * is simply a Container with added training routines.
 *
 */
export class Container extends Layer {
    constructor(args) {
        // No args passed to super's constructor.
        super({});
        this.containerNodes = new Set();
        this.name = args.name;
        if (this.name == null) {
            const prefix = this.getClassName().toLowerCase();
            this.name = getUid(prefix);
        }
        this.supportsMasking = false;
        this.trainable_ = true;
        // TODO(michaelterry): Initialize perInputLosses/Updates here.
        // Container-specific properties.
        if (Array.isArray(args.inputs)) {
            this.inputs = args.inputs.slice();
        }
        else {
            this.inputs = [args.inputs];
        }
        if (Array.isArray(args.outputs)) {
            this.outputs = args.outputs.slice();
        }
        else {
            this.outputs = [args.outputs];
        }
        // Check for redundancy in inputs.
        if (generic_utils.unique(this.inputs).length !== this.inputs.length) {
            throw new ValueError('The list of inputs passed to the model is ' +
                'redundant. All inputs should only appear once. Found: ' +
                `${this.inputs.map(x => x.name)}`);
        }
        // Check for redundancy in outputs.
        if (generic_utils.unique(this.outputs).length !== this.outputs.length) {
            console.warn('The list of outputs passed to the model is redundant. ' +
                'All outputs should only appear once. Found: ' +
                `${this.outputs.map(x => x.name)}`);
        }
        /*
          List of initial layers (1 to 1 mapping with this.inputs, hence the same
          layer might appear twice)
        */
        this.inputLayers = [];
        this.inputLayersNodeIndices = [];
        this.inputLayersTensorIndices = [];
        /*
          List of layers (1 to 1 mapping with this.outputs, hence the same layer
          might appear twice)
        */
        this.outputLayers = [];
        this.outputLayersNodeIndices = [];
        this.outputLayersTensorIndices = [];
        /*
          All layers in order of horizontal graph traversal. Entries are unique.
          Includes input and output layers.
        */
        this.layers = [];
        /*
          References to container layers that were constructed internally. We need
          these to properly dispose of tensors from nested containers.
        */
        this.internalContainerRefs = [];
        // TODO(michaelterry): Determine if caching still needed with eager
        // backend.
        /*
          This is for performance optimization when calling the Container on new
          inputs. Every time the Container is called on a set on input tensors,
          we compute the output tensors, output masks and output shapes in one pass,
          then cache them here. When one of these outputs is queried later,
          we retrieve it from there instead of recomputing it.
        */
        // this.outputTensorCache = {};
        // this.outputShapeCache = {};
        // Build this.outputLayers:
        for (const x of this.outputs) {
            const layer = x.sourceLayer;
            const nodeIndex = x.nodeIndex;
            const tensorIndex = x.tensorIndex;
            this.outputLayers.push(layer);
            this.outputLayersNodeIndices.push(nodeIndex);
            this.outputLayersTensorIndices.push(tensorIndex);
        }
        // TODO(michaelterry): Add output mask cache code.
        // Build this.inputLayers:
        for (const x of this.inputs) {
            const layer = x.sourceLayer;
            const nodeIndex = x.nodeIndex;
            const tensorIndex = x.tensorIndex;
            /*
              It's supposed to be an input layer, so only one node
              and one tensor output.
            */
            generic_utils.assert(nodeIndex === 0, 'input layer has >1 nodes');
            generic_utils.assert(tensorIndex === 0, 'input layer has >1 tensors');
            this.inputLayers.push(layer);
            this.inputLayersNodeIndices.push(nodeIndex);
            this.inputLayersTensorIndices.push(tensorIndex);
        }
        // Build this.inputNames and this.outputNames.
        this.inputNames = [];
        this.outputNames = [];
        this.feedInputShapes = [];
        this.feedInputNames = [];
        this.feedOutputNames = [];
        for (let i = 0; i < this.inputLayers.length; i++) {
            const layer = this.inputLayers[i];
            // Check that layer is an InputLayer.
            if (!(layer instanceof InputLayer)) {
                throw new TypeError('Input layers to a LayersModel must be InputLayer objects. ' +
                    `Received inputs: ${args.inputs}. ` +
                    `Input ${i} (0-based) originates ` +
                    `from layer type ${layer.getClassName()}.`);
            }
            this.inputNames.push(layer.name);
            this.feedInputShapes.push(layer.batchInputShape);
            this.feedInputNames.push(layer.name);
        }
        for (const layer of this.outputLayers) {
            this.outputNames.push(layer.name);
        }
        this.internalInputShapes = this.inputs.map(x => x.shape);
        this.internalOutputShapes = this.outputs.map(x => x.shape);
        /*
          Container_nodes: set of nodes included in the graph (not all nodes
          included in the layers are relevant to the current graph).
        */
        // ids of all nodes relevant to the Container:
        const nodesDepths = {};
        // To recover nodes from their ID.
        const nodeIDToNode = {};
        const layersDepths = {};
        // To layers from their ID.
        const layerIDToLayer = {};
        const layerIndices = {};
        const nodesInDecreasingDepth = [];
        /**
         * Builds a map of the graph of layers.
         *
         * This recursively updates the map `layerIndices`,
         * the list `nodesInDecreasingDepth` and the set `containerNodes`.
         *
         * @param tensor Some tensor in a graph.
         * @param finishedNodes Set of nodes whose subgraphs have been traversed
         *         completely. Useful to prevent duplicated work.
         * @param nodesInProgress Set of nodes that are currently active on the
         *         recursion stack. Useful to detect cycles.
         * @param layer Layer from which `tensor` comes from. If not provided,
         *   will be obtained from tensor.sourceLayer.
         * @param nodeIndex Node index from which `tensor` comes from.
         * @param tensorIndex TensorIndex from which `tensor` comes from.
         *
         * @exception RuntimeError if a cycle is detected.
         */
        const buildMapOfGraph = (tensor, finishedNodes, nodesInProgress, layer, nodeIndex, tensorIndex) => {
            if (layer == null || nodeIndex == null || tensorIndex == null) {
                layer = tensor.sourceLayer;
                nodeIndex = tensor.nodeIndex;
                tensorIndex = tensor.tensorIndex;
            }
            const node = layer.inboundNodes[nodeIndex];
            // Prevent cycles.
            if (nodesInProgress.indexOf(node) !== -1) {
                throw new RuntimeError(`The tensor ${tensor.name} at layer "${layer.name}" ` +
                    'is part of a cycle.');
            }
            // Don't repeat work for shared subgraphs
            if (finishedNodes.indexOf(node) !== -1) {
                return;
            }
            // Update containerNodes.
            this.containerNodes.add(Container.nodeKey(layer, nodeIndex));
            // Store the traversal order for layer sorting.
            if (!(layer.id in layerIndices)) {
                layerIndices[layer.id] = Object.keys(layerIndices).length;
            }
            if (nodesInProgress.indexOf(node) === -1) {
                nodesInProgress.push(node);
            }
            // Propagate to all previous tensors connected to this node.
            const numInboundLayers = node.inboundLayers.length;
            for (let i = 0; i < numInboundLayers; i++) {
                const x = node.inputTensors[i];
                const layer = node.inboundLayers[i];
                const nodeIndex = node.nodeIndices[i];
                const tensorIndex = node.tensorIndices[i];
                buildMapOfGraph(x, finishedNodes, nodesInProgress, layer, nodeIndex, tensorIndex);
            }
            finishedNodes.push(node);
            while (nodesInProgress.indexOf(node) >= 0) {
                nodesInProgress.splice(nodesInProgress.indexOf(node), 1);
            }
            nodesInDecreasingDepth.push(node);
        };
        const finishedNodes = [];
        const nodesInProgress = [];
        for (const x of this.outputs) {
            buildMapOfGraph(x, finishedNodes, nodesInProgress);
        }
        const reversedNodesInDecreasingDepth = nodesInDecreasingDepth.slice().reverse();
        for (const node of reversedNodesInDecreasingDepth) {
            nodeIDToNode[node.id] = node;
            // If the depth is not set, the node has no outbound nodes (depth 0).
            if (!(node.id in nodesDepths)) {
                nodesDepths[node.id] = 0;
            }
            let depth = nodesDepths[node.id];
            // Update the depth of the corresponding layer
            const previousDepth = (layersDepths[node.outboundLayer.id] == null ?
                0 :
                layersDepths[node.outboundLayer.id]);
            /*
              If we've seen this layer before at a higher depth, we should use that
              depth instead of the node depth.  This is necessary for shared layers
              that have inputs at different depth levels in the graph.
            */
            depth = Math.max(depth, previousDepth);
            layersDepths[node.outboundLayer.id] = depth;
            layerIDToLayer[node.outboundLayer.id] = node.outboundLayer;
            nodesDepths[node.id] = depth;
            // Update the depth of inbound nodes.
            for (let i = 0; i < node.inboundLayers.length; i++) {
                const inboundLayer = node.inboundLayers[i];
                const nodeIndex = node.nodeIndices[i];
                const inboundNode = inboundLayer.inboundNodes[nodeIndex];
                const previousDepth = (nodesDepths[inboundNode.id] == null ? 0 :
                    nodesDepths[inboundNode.id]);
                nodesDepths[inboundNode.id] = Math.max(depth + 1, previousDepth);
                nodeIDToNode[inboundNode.id] = inboundNode;
            }
        }
        // Build a dict {depth: list of nodes with this depth}
        const nodesByDepth = {};
        for (const nodeID in nodesDepths) {
            const depth = nodesDepths[nodeID];
            if (!(depth in nodesByDepth)) {
                nodesByDepth[depth] = [];
            }
            nodesByDepth[depth].push(nodeIDToNode[nodeID]);
        }
        // Build a dict {depth: list of layers with this depth}
        const layersByDepth = {};
        for (const layerID in layersDepths) {
            const depth = layersDepths[layerID];
            if (!(depth in layersByDepth)) {
                layersByDepth[depth] = [];
            }
            layersByDepth[depth].push(layerIDToLayer[layerID]);
        }
        // Get sorted list of layer depths.
        let depthKeys = Object.keys(layersByDepth)
            .map(x => parseInt(x, 10))
            .sort(generic_utils.reverseNumberCompare);
        // Set this.layers and this.layersByDepth.
        this.layers = [];
        for (const depth of depthKeys) {
            const layersForDepth = layersByDepth[depth];
            // Container.layers needs to have a deterministic order:
            // here we order them by traversal order.
            layersForDepth.sort((a, b) => {
                const aIndex = layerIndices[a.id];
                const bIndex = layerIndices[b.id];
                if (aIndex < bIndex) {
                    return -1;
                }
                if (aIndex > bIndex) {
                    return 1;
                }
                return 0;
            });
            for (const layer of layersForDepth) {
                if (layer instanceof Container) {
                    this.internalContainerRefs.push(layer);
                }
                this.layers.push(layer);
            }
        }
        this.layersByDepth = layersByDepth;
        // Get sorted list of node depths;
        depthKeys = Object.keys(nodesByDepth)
            .map(x => parseInt(x, 10))
            .sort(generic_utils.reverseNumberCompare);
        // Check that all tensors required are computable.
        // computable_tensors: all tensors in the graph
        // that can be computed from the inputs provided.
        const computableTensors = this.inputs.slice();
        // To provide a better error msg.
        const layersWithCompleteInput = [];
        for (const depth of depthKeys) {
            for (const node of nodesByDepth[depth]) {
                const layer = node.outboundLayer;
                if (layer != null) {
                    for (const x of node.inputTensors) {
                        if (computableTensors.indexOf(x) === -1) {
                            throw new RuntimeError(`Graph disconnected: cannot obtain value for tensor ${x}` +
                                ` at layer "${layer.name}". ` +
                                'The following previous layers were accessed without ' +
                                `issue: ${layersWithCompleteInput}`);
                        }
                    }
                    for (const x of node.outputTensors) {
                        computableTensors.push(x);
                    }
                    layersWithCompleteInput.push(layer.name);
                }
            }
        }
        // Set this.containerNodes and this.nodesByDepth.
        this.nodesByDepth = nodesByDepth;
        // Ensure name unicity, which will be crucial for serialization
        // (since serialized nodes refer to layers by their name).
        const allNames = this.layers.map(x => x.name);
        for (const name of allNames) {
            const numOccurrences = allNames.filter(x => x === name).length;
            if (numOccurrences !== 1) {
                throw new RuntimeError(`The name "${name}" is used ${numOccurrences} times ` +
                    'in the model. All layer names should be unique. Layer names: ' +
                    JSON.stringify(allNames));
            }
        }
        // Layer parameters.
        // The new container starts with a single inbound node
        // for its inputs, and no outbound nodes.
        // Will be appended to by future calls to apply().
        this.outboundNodes = [];
        // Will be appended to below, and by future calls to apply().
        this.inboundNodes = [];
        // Create the node linking internal inputs to internal outputs.
        // (This call has side effects.)
        // tslint:disable-next-line:no-unused-expression
        new Node({
            outboundLayer: this,
            inboundLayers: [],
            nodeIndices: [],
            tensorIndices: [],
            inputTensors: this.inputs,
            outputTensors: this.outputs,
            inputMasks: this.inputs.map(x => null),
            outputMasks: this.outputs.map(x => null),
            inputShapes: this.inputs.map(x => x.shape),
            outputShapes: this.outputs.map(x => x.shape)
        });
        this.built = true;
        this._refCount = 1; // The ref count of a container always start at 1.
    }
    assertNotDisposed() {
        if (this._refCount === 0) {
            throw new Error(`Container '${this.name}' is already disposed.`);
        }
    }
    /**
     * Attempt to dispose a LayersModel's weights.
     *
     * This method decrease the reference count of the LayersModel object by 1.
     *
     * A LayersModel is reference-counted. Its reference count is incremented by 1
     * when it is first constructed and when it is used as a Layer of another
     * LayersModel.
     *
     * If the reference count of a LayersModel becomes 0, the `dispose` method of
     * all its constituent `Layer`s will be called.
     *
     * Note: If the reference count is greater than 0 after the decrement, the
     * `dispose` method of its constituent `Layer`s will *not* be called.
     *
     * After a LayersModel is disposed, it cannot be used in calls such as
     * 'predict`, `evaluate` or `fit` anymore.
     *
     * @returns A DisposeResult Object with the following fields:
     *   - refCountAfterDispose: The reference count of the LayersModel after this
     *     `dispose()` call.
     *   - numDisposedVariables: Number of `tf.Variable`s (i.e., weights) disposed
     *     during this `dispose()` call.
     * @throws {Error} If the layer is not built yet, or if the LayersModel has
     *   already been disposed.
     */
    dispose() {
        this.assertNotDisposed();
        const result = { refCountAfterDispose: null, numDisposedVariables: 0 };
        if (--this._refCount === 0) {
            for (const layer of this.layers) {
                result.numDisposedVariables += layer.dispose().numDisposedVariables;
            }
            // Call dispose on each internally created container layer again to ensure
            // their refCounts hit zero and their tensors are subsequently deleted.
            for (const container of this.internalContainerRefs) {
                result.numDisposedVariables += container.dispose().numDisposedVariables;
            }
        }
        result.refCountAfterDispose = this._refCount;
        return result;
    }
    get trainable() {
        return this.trainable_;
    }
    set trainable(trainable) {
        this.layers.forEach(layer => {
            // tslint:disable-next-line:no-any
            layer._trainableWeights
                .forEach(w => w.trainable = trainable);
        });
        this.trainable_ = trainable;
    }
    get trainableWeights() {
        // Porting Note: This check below is to prevent errors where the
        //   _trainableWeights inherited from the parent class (Layer) gets
        //   inadvertently used.
        if (this._trainableWeights.length > 0) {
            throw new ValueError('Container instance unexpectedly contains _trainableWeights.' +
                'The trainable weights of a Container are a union of the ' +
                'trainable weights of its consituent Layers. Its own ' +
                '_trainableWeights must remain an empty Array.');
        }
        if (!this.trainable) {
            return [];
        }
        let weights = [];
        for (const layer of this.layers) {
            weights = weights.concat(layer.trainableWeights);
        }
        return weights;
    }
    get nonTrainableWeights() {
        const weights = [];
        for (const layer of this.layers) {
            weights.push(...layer.nonTrainableWeights);
        }
        if (!this.trainable) {
            const trainableWeights = [];
            for (const layer of this.layers) {
                trainableWeights.push(...layer.trainableWeights);
            }
            return trainableWeights.concat(weights);
        }
        return weights;
    }
    get weights() {
        return this.trainableWeights.concat(this.nonTrainableWeights);
    }
    /**
     * Loads all layer weights from a JSON object.
     *
     * Porting Note: HDF5 weight files cannot be directly loaded in JavaScript /
     *   TypeScript. The utility script at `scripts/pykeras.py` offers means
     *   to convert them into JSON strings compatible with this method.
     * Porting Note: TensorFlow.js Layers supports only loading by name currently.
     *
     * @param weights A JSON mapping weight names to weight values as nested
     *   arrays of numbers, or a `NamedTensorMap`, i.e., a JSON mapping weight
     *   names to `tf.Tensor` objects.
     * @param strict Require that the provided weights exactly match those
     *   required by the container.  Default: `true`.  Passing `false` means that
     *   extra weights and missing weights will be silently ignored.
     */
    loadWeights(weights, strict = true) {
        const nameToWeight = {};
        let totalWeightsCount = 0;
        for (const layer of this.layers) {
            for (const weight of layer.weights) {
                if (nameToWeight[weight.originalName] != null) {
                    throw new ValueError(`Duplicate weight name: ${weight.originalName}`);
                }
                nameToWeight[weight.originalName] = weight;
                totalWeightsCount++;
            }
        }
        const weightValueTuples = [];
        for (const name in weights) {
            // TF 2.2.0 added cell name to the weight name in the format of
            // layer_name/cell_name/weight_name, we need to remove
            // the inner cell name.
            let validatedName = name;
            if (nameToWeight[name] == null) {
                const tokens = name.split('/');
                const shortenNameArray = tokens.slice(0, -2).concat([tokens[tokens.length - 1]]);
                validatedName = shortenNameArray.join('/');
            }
            if (nameToWeight[validatedName] != null) {
                weightValueTuples.push([nameToWeight[validatedName], weights[name]]);
            }
            else if (strict) {
                throw new ValueError(`Provided weight data has no target variable: ${name}`);
            }
            delete nameToWeight[validatedName];
        }
        if (strict) {
            // Check that all weights are set.
            const unsetNames = [];
            for (const name in nameToWeight) {
                unsetNames.push(name);
            }
            if (unsetNames.length > 0) {
                throw new ValueError(`${unsetNames.length} of ${totalWeightsCount} weights are not set: ` +
                    `${unsetNames}`);
            }
        }
        batchSetValue(weightValueTuples);
    }
    /**
     * Util shared between different serialization methods.
     * @returns LayersModel config with Keras version information added.
     */
    updatedConfig() {
        const theConfig = this.getConfig();
        const modelConfig = {};
        modelConfig['className'] = this.getClassName();
        modelConfig['config'] = theConfig;
        modelConfig['kerasVersion'] = `tfjs-layers ${layersVersion}`;
        // TODO(nielsene): Replace something like K.backend() once
        // possible.
        modelConfig['backend'] = 'TensorFlow.js';
        return modelConfig;
    }
    /**
     * Returns a JSON string containing the network configuration.
     *
     * To load a network from a JSON save file, use
     * models.modelFromJSON(jsonString);
     * @param extraJsonArgs Unused in tfjs-layers, maintained for PyKeras
     * @param returnString Whether the return value should be stringified
     *    (default: `true`).
     * @returns a JSON string if `returnString` (default), or a JSON object if
     *   `!returnString`.
     */
    // tslint:disable-next-line:no-any
    toJSON(unused, returnString = true) {
        const modelConfig = convertTsToPythonic(this.updatedConfig());
        return returnString ? JSON.stringify(modelConfig) : modelConfig;
    }
    /**
     * Call the model on new inputs.
     *
     * In this case `call` just reapplies all ops in the graph to the new inputs
     * (e.g. build a new computational graph from the provided inputs).
     *
     * @param inputs A tensor or list of tensors.
     * @param mask A mask or list of masks. A mask can be either a tensor or null
     *   (no mask).
     *
     * @return A tensor if there is a single output, or a list of tensors if there
     *   are more than one outputs.
     */
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = generic_utils.toList(inputs);
            const feedDict = new FeedDict();
            for (let i = 0; i < this.inputs.length; ++i) {
                feedDict.add(this.inputs[i], inputs[i]);
            }
            return execute(this.outputs, feedDict, kwargs);
        });
    }
    /**
     * Computes an output mask tensor.
     *
     * @param inputs Tensor or list of tensors.
     * @param mask Tensor or list of tensors.
     *
     * @return null or a tensor (or list of tensors, one per output tensor of the
     * layer).
     */
    computeMask(inputs, mask) {
        return tidy(() => {
            inputs = generic_utils.toList(inputs);
            let masks;
            if (mask == null) {
                masks = generic_utils.pyListRepeat(null, inputs.length);
            }
            else {
                masks = generic_utils.toList(mask);
            }
            // TODO(michaelterry): Add support for mask caching.
            return this.runInternalGraph(inputs, masks)[1];
        });
    }
    /**
     * Computes the output shape of the layer.
     *
     * Assumes that the layer will be built to match that input shape provided.
     *
     * @param inputShape A shape (tuple of integers) or a list of shape tuples
     *   (one per output tensor of the layer). Shape tuples can include null for
     *   free dimensions, instead of an integer.
     */
    computeOutputShape(inputShape) {
        const inputShapes = types_utils.normalizeShapeList(inputShape);
        if (inputShapes.length !== this.inputLayers.length) {
            throw new ValueError(`Invalid inputShape argument ${inputShape}: ` +
                `model has ${this.inputLayers.length} tensor inputs.`);
        }
        // TODO(michaelterry): Add caching
        const layersToOutputShapes = {};
        for (let i = 0; i < inputShapes.length; i++) {
            const layer = this.inputLayers[i];
            const inputShape = inputShapes[i];
            // It's an input layer: computeOutputShape is identity,
            // and there is only one node and one tensor output.
            const shapeKey = layer.name + '_0_0';
            layersToOutputShapes[shapeKey] = inputShape;
        }
        const depthKeys = Object.keys(this.nodesByDepth)
            .map(x => parseInt(x, 10))
            .sort(generic_utils.reverseNumberCompare);
        // Iterate over nodes, by depth level.
        if (depthKeys.length > 1) {
            for (const depth of depthKeys) {
                const nodes = this.nodesByDepth[depth];
                for (const node of nodes) {
                    // This is always a single layer, never a list.
                    const layer = node.outboundLayer;
                    if (this.inputLayers.map(x => x.id).indexOf(layer.id) !== -1) {
                        // We've already covered the input layers a few lines above.
                        continue;
                    }
                    // Potentially redundant list, same size of node.inputTensors.
                    const inputShapes = [];
                    for (let j = 0; j < node.inboundLayers.length; j++) {
                        const inboundLayer = node.inboundLayers[j];
                        const nodeIndex = node.nodeIndices[j];
                        const tensorIndex = node.tensorIndices[j];
                        const shapeKey = `${inboundLayer.name}_${nodeIndex}_${tensorIndex}`;
                        const inputShape = layersToOutputShapes[shapeKey];
                        inputShapes.push(inputShape);
                    }
                    const outputShape = layer.computeOutputShape(generic_utils.singletonOrArray(inputShapes));
                    const outputShapes = types_utils.normalizeShapeList(outputShape);
                    const nodeIndex = layer.inboundNodes.indexOf(node);
                    for (let j = 0; j < outputShapes.length; j++) {
                        const shapeKey = `${layer.name}_${nodeIndex}_${j}`;
                        layersToOutputShapes[shapeKey] = outputShapes[j];
                    }
                }
            }
        }
        // Read final output shapes from layersToOutputShapes.
        const outputShapes = [];
        const outputShapeKeys = [];
        for (let i = 0; i < this.outputLayers.length; i++) {
            const layer = this.outputLayers[i];
            const nodeIndex = this.outputLayersNodeIndices[i];
            const tensorIndex = this.outputLayersTensorIndices[i];
            const shapeKey = `${layer.name}_${nodeIndex}_${tensorIndex}`;
            outputShapeKeys.push(shapeKey);
        }
        for (let i = 0; i < outputShapeKeys.length; i++) {
            const key = outputShapeKeys[i];
            generic_utils.assert(key in layersToOutputShapes);
            outputShapes.push(layersToOutputShapes[key]);
        }
        // TODO(michaelterry): Update cache
        return generic_utils.singletonOrArray(outputShapes);
    }
    /**
     * Computes output tensors for new inputs.
     *
     * Note:
     *   - Expects `inputs` to be a list (potentially with 1 element).
     *
     * @param inputs List of tensors
     * @param masks List of masks (tensors or null).
     * @return Three lists: outputTensors, outputMasks, outputShapes
     */
    runInternalGraph(inputs, masks) {
        if (masks == null) {
            masks = generic_utils.pyListRepeat(null, inputs.length);
        }
        // Dictionary mapping reference tensors to tuples
        // (computed tensor, compute mask)
        // we assume a 1:1 mapping from tensor to mask
        // TODO: raise exception when a `.computeMask()` call
        // does not return a list the same size as `call`
        const tensorMap = {};
        for (let i = 0; i < this.inputs.length; ++i) {
            const x = this.inputs[i];
            const y = inputs[i];
            const mask = masks[i];
            tensorMap[x.id] = [y, mask];
        }
        const depthKeys = Object.keys(this.nodesByDepth)
            .map(x => parseInt(x, 10))
            .sort(generic_utils.reverseNumberCompare);
        for (const depth of depthKeys) {
            const nodes = this.nodesByDepth[depth];
            for (const node of nodes) {
                // This is always a single layer, never a list.
                const layer = node.outboundLayer;
                const referenceInputTensors = node.inputTensors;
                const referenceOutputTensors = node.outputTensors;
                // If all previous input tensors are available in tensorMap,
                // then call node.inboundLayer on them.
                // List of tuples [input, mask]:
                const computedData = new Array();
                for (const x of referenceInputTensors) {
                    if (x.id in tensorMap) {
                        computedData.push(tensorMap[x.id]);
                    }
                }
                if (computedData.length === referenceInputTensors.length) {
                    // TODO(michaelterry): Add K.name_scope here, if we need it.
                    let kwargs = {};
                    let computedTensors;
                    let computedMasks;
                    let outputTensors;
                    let outputMasks;
                    // call layer
                    if (node.callArgs != null) {
                        kwargs = node.callArgs;
                    }
                    if (computedData.length === 1) {
                        const [computedTensor, computedMask] = computedData[0];
                        if (kwargs['mask'] == null) {
                            kwargs['mask'] = computedMask;
                        }
                        outputTensors =
                            generic_utils.toList(layer.call(computedTensor, kwargs));
                        outputMasks = generic_utils.toList(layer.computeMask(computedTensor, computedMask));
                        computedTensors = [computedTensor];
                        computedMasks = [computedMask];
                    }
                    else {
                        computedTensors = computedData.map(x => x[0]);
                        computedMasks = computedData.map(x => x[1]);
                        if (kwargs['mask'] == null) {
                            kwargs['mask'] = computedMasks;
                        }
                        outputTensors =
                            generic_utils.toList(layer.call(computedTensors, kwargs));
                        outputMasks = generic_utils.toList(layer.computeMask(computedTensors, computedMasks));
                    }
                    if (layer.activityRegularizer) {
                        throw new NotImplementedError('LayersModel invocation with concrete Tensor value(s) in the ' +
                            'presence of activity regularizer(s) is not supported yet.');
                    }
                    // TODO(michaelterry): Add model updates and losses
                    // Update tensor map.
                    for (let i = 0; i < referenceOutputTensors.length; ++i) {
                        const x = referenceOutputTensors[i];
                        const y = outputTensors[i];
                        const mask = outputMasks[i];
                        tensorMap[x.id] = [y, mask];
                    }
                }
            }
        }
        const outputTensors = [];
        const outputMasks = [];
        const outputShapes = [];
        for (const x of this.outputs) {
            generic_utils.assert(x.id in tensorMap, `Could not compute output ${x.name} : ${x.id}`);
            const [tensor, mask] = tensorMap[x.id];
            outputShapes.push(tensor.shape);
            outputTensors.push(tensor);
            outputMasks.push(mask);
        }
        // TODO(michaelterry): Add support for caches.
        return [outputTensors, outputMasks, outputShapes];
    }
    /**
     * Builds a map of internal node keys to node ordering.
     * Used in serializaion a node orderings may change as unused nodes are
     * dropped. Porting Note:  This helper method was pulled out of getConfig to
     * improve readability.
     * @param layers An array of Layers in the model.
     * @returns Map of Node Keys to index order within the layer.
     */
    buildNodeConversionMap(layers) {
        const nodeConversionMap = {};
        let keptNodes;
        for (const layer of this.layers) {
            keptNodes = layer instanceof Container ? 1 : 0;
            for (let originalNodeIndex = 0; originalNodeIndex < layer.inboundNodes.length; originalNodeIndex++) {
                const nodeKey = Container.nodeKey(layer, originalNodeIndex);
                if (this.containerNodes.has(nodeKey)) {
                    // i.e. we mark it to be saved
                    nodeConversionMap[nodeKey] = keptNodes;
                    keptNodes += 1;
                }
            }
        }
        return nodeConversionMap;
    }
    /**
     * Retrieves a layer based on either its name (unique) or index.
     *
     * Indices are based on order of horizontal graph traversal (bottom-up).
     *
     * If both `name` and `index` are specified, `index` takes precedence.
     *
     * @param name Name of layer.
     * @param index Index of layer.
     * @returns A Layer instance.
     * @throws ValueError: In case of invalid layer name or index.
     *
     * @doc {
     *    heading: 'Layers',
     *    subheading: 'Classes',
     *    namespace: 'layers',
     *    subclasses: ['LayersModel']
     * }
     */
    getLayer(name, index) {
        if (index != null) {
            if (this.layers.length <= index) {
                throw new ValueError(`Was asked to retrieve layer at index ${index}, but model only ` +
                    `has ${this.layers.length} layer(s).`);
            }
            else {
                return this.layers[index];
            }
        }
        else {
            if (name == null) {
                throw new ValueError('Provide either a layer name or layer index');
            }
        }
        for (const layer of this.layers) {
            if (layer.name === name) {
                return layer;
            }
        }
        throw new ValueError(`No such layer: ${name}`);
    }
    /**
     * Retrieves the Container's current loss values.
     *
     * Used for regularizers during training.
     */
    calculateLosses() {
        // Porting Node: This is an augmentation to Container.loss in PyKeras.
        //   In PyKeras, Container.loss returns symbolic tensors. Here a concrete
        //   Tensor (specifically Scalar) values are returned. This is due to the
        //   imperative backend.
        return tidy(() => {
            const losses = [];
            for (const layer of this.layers) {
                for (let nodeIndex = 0; nodeIndex < layer.inboundNodes.length; ++nodeIndex) {
                    const nodeKey = Container.nodeKey(layer, nodeIndex);
                    if (this.containerNodes.has(nodeKey)) {
                        losses.push(...layer.calculateLosses());
                    }
                }
            }
            // TODO(cais): Add any unconditional model-level losses?
            return losses;
        });
    }
    getConfig() {
        const config = { name: this.name };
        // Build a map from layer unique name (self._node_key)
        // to the index of the nodes that are saved in the config.
        // Only nodes in container_nodes are saved.
        const nodeConversionMap = this.buildNodeConversionMap(this.layers);
        // Serialize and save the layers in layerConfigs
        const layerConfigs = [];
        for (const layer of this.layers) {
            const layerClassName = layer.getClassName();
            const layerConfig = layer.getConfig();
            const filteredInboundNodes = [];
            for (let originalNodeIndex = 0; originalNodeIndex < layer.inboundNodes.length; originalNodeIndex++) {
                const node = layer.inboundNodes[originalNodeIndex];
                const nodeKey = Container.nodeKey(layer, originalNodeIndex);
                let kwargs = {};
                if (this.containerNodes.has(nodeKey)) {
                    // The node is relevant to the model:
                    // add to filteredInboundNodes.
                    if (node.callArgs) {
                        try {
                            JSON.stringify(node.callArgs);
                            kwargs = node.callArgs;
                        }
                        catch (err) {
                            console.warn(`Layer ${layer.name} was passed ` +
                                `non-serializable keyword arguments: ` +
                                `${node.callArgs}. They will not be included ` +
                                `in the serialized model (and thus will be ` +
                                `missing at deserialization time).`);
                            kwargs = {};
                        }
                    }
                    if (node.inboundLayers.length > 0) {
                        const nodeData = [];
                        for (let i = 0; i < node.inboundLayers.length; i++) {
                            const inboundLayer = node.inboundLayers[i];
                            const nodeIndex = node.nodeIndices[i];
                            const tensorIndex = node.tensorIndices[i];
                            const nodeKey = Container.nodeKey(inboundLayer, nodeIndex);
                            let newNodeIndex = nodeConversionMap[nodeKey];
                            if (newNodeIndex == null) {
                                newNodeIndex = 0;
                            }
                            nodeData.push([inboundLayer.name, newNodeIndex, tensorIndex, kwargs]);
                        }
                        filteredInboundNodes.push(nodeData);
                    }
                }
            }
            const dict = {};
            dict['name'] = layer.name;
            dict['className'] = layerClassName;
            dict['config'] = layerConfig;
            dict['inboundNodes'] = filteredInboundNodes;
            layerConfigs.push(dict);
        }
        config['layers'] = layerConfigs;
        // Gather info about inputs and outputs
        const modelInputs = [];
        for (let i = 0; i < this.inputLayers.length; i++) {
            const layer = this.inputLayers[i];
            const nodeIndex = this.inputLayersNodeIndices[i];
            const nodeKey = Container.nodeKey(layer, nodeIndex);
            if (!this.containerNodes.has(nodeKey)) {
                continue;
            }
            let newNodeIndex = nodeConversionMap[nodeKey];
            if (newNodeIndex === null || newNodeIndex === undefined) {
                newNodeIndex = 0;
            }
            const tensorIndex = this.inputLayersTensorIndices[i];
            modelInputs.push([layer.name, newNodeIndex, tensorIndex]);
        }
        config['inputLayers'] = modelInputs;
        const modelOutputs = [];
        for (let i = 0; i < this.outputLayers.length; i++) {
            const layer = this.outputLayers[i];
            const nodeIndex = this.outputLayersNodeIndices[i];
            const nodeKey = Container.nodeKey(layer, nodeIndex);
            if (!this.containerNodes.has(nodeKey)) {
                continue;
            }
            let newNodeIndex = nodeConversionMap[nodeKey];
            if (newNodeIndex === null || newNodeIndex === undefined) {
                newNodeIndex = 0;
            }
            const tensorIndex = this.outputLayersTensorIndices[i];
            modelOutputs.push([layer.name, newNodeIndex, tensorIndex]);
        }
        config['outputLayers'] = modelOutputs;
        return config;
    }
    /**
     * Instantiates a LayersModel from its config (output of `get_config()`).
     * @param cls the class to create
     * @param config LayersModel config dictionary.
     * @param customObjects An optional dictionary of custom objects.
     * @param fastWeightInit Optional flag to use fast weight initialization
     *   during deserialization. This is applicable to cases in which
     *   the initialization will be immediately overwritten by loaded weight
     *   values. Default: `false`.
     * @returns A LayersModel instance.
     * @throws ValueError: In case of improperly formatted config dict.
     */
    /** @nocollapse */
    static fromConfig(cls, config, customObjects = {}, fastWeightInit = false) {
        // Layer instances created during
        // the graph reconstruction process
        const createdLayers = {};
        // Dictionary mapping layer instances to
        // node data that specifies a layer call.
        // It acts as a queue that maintains any unprocessed
        // layer call until it becomes possible to process it
        // (i.e. until the input tensors to the call all exist).
        const unprocessedNodes = {};
        function addUnprocessedNode(layer, nodeData) {
            if (!(layer.name in unprocessedNodes)) {
                unprocessedNodes[layer.name] = [nodeData];
            }
            else {
                unprocessedNodes[layer.name].push(nodeData);
            }
        }
        function processNode(layer, nodeData) {
            const inputTensors = [];
            let kwargs;
            for (const inputData of nodeData) {
                const inboundLayerName = inputData[0];
                const inboundNodeIndex = inputData[1];
                const inboundTensorIndex = inputData[2];
                kwargs = inputData[3] == null ?
                    {} :
                    inputData[3];
                if (!(inboundLayerName in createdLayers)) {
                    addUnprocessedNode(layer, nodeData);
                    return;
                }
                const inboundLayer = createdLayers[inboundLayerName];
                if (inboundLayer.inboundNodes.length <= inboundNodeIndex) {
                    addUnprocessedNode(layer, nodeData);
                    return;
                }
                const inboundNode = inboundLayer.inboundNodes[inboundNodeIndex];
                inputTensors.push(inboundNode.outputTensors[inboundTensorIndex]);
            }
            // Call layer on its inputs, thus creating the node
            // and building the layer if needed.
            // Note: This has Eager vs Graph Implications.
            if (inputTensors.length > 0) {
                layer.apply(generic_utils.singletonOrArray(inputTensors), kwargs); // was ** kwargs
            }
        }
        /**
         * Deserialize a layer, then call it on appropriate inputs.
         * @param layerData: layer config dict.
         * @throws ValueError: In case of improperly formatted `layer_data`
         * dict.
         */
        function processLayer(layerData) {
            const layerName = layerData['name'];
            // Instantiate layer.
            const layer = deserializeLayer(layerData, config['customObjects'] != null ?
                config['customObjects'] :
                {});
            layer.setFastWeightInitDuringBuild(fastWeightInit);
            createdLayers[layerName] = layer;
            // Gather layer inputs.
            const inboundNodesData = layerData['inboundNodes'];
            inboundNodesData.forEach(nodeData => {
                if (!(nodeData instanceof Array)) {
                    throw new ValueError(`Corrupted configuration, expected array for nodeData: ${nodeData}`);
                }
                // We don't process nodes (i.e. make layer calls)
                // on the fly because the inbound node may not yet exist,
                // in case of layer shared at different topological depths
                // (e.g.a model such as A(B(A(B(x)))))
                addUnprocessedNode(layer, nodeData);
            });
        }
        // First, we create all layers and enqueue nodes to be processed.
        const name = config['name'];
        const layersFromConfig = config['layers'];
        for (const layerData of layersFromConfig) {
            processLayer(layerData);
        }
        // Then we process nodes in order of layer depth.
        // Nodes that cannot yet be processed(if the inbound node
        // does not yet exist) are re - enqueued, and the process
        // is repeated until all nodes are processed.
        while (!generic_utils.isObjectEmpty(unprocessedNodes)) {
            for (const layerData of layersFromConfig) {
                const layer = createdLayers[layerData['name']];
                if (layer.name in unprocessedNodes) {
                    const currentUnprocessedNodesForLayer = unprocessedNodes[layer.name];
                    delete unprocessedNodes[layer.name];
                    for (const nodeData of currentUnprocessedNodesForLayer) {
                        processNode(layer, nodeData);
                    }
                }
            }
        }
        const inputTensors = [];
        const outputTensors = [];
        const inputLayersFromConfig = config['inputLayers'];
        for (const layerData of inputLayersFromConfig) {
            const layerName = layerData[0];
            const nodeIndex = layerData[1];
            const tensorIndex = layerData[2];
            generic_utils.assert(layerName in createdLayers);
            const layer = createdLayers[layerName];
            const layerOutputTensors = layer.inboundNodes[nodeIndex].outputTensors;
            inputTensors.push(layerOutputTensors[tensorIndex]);
        }
        const outputLayersFromConfig = config['outputLayers'];
        for (const layerData of outputLayersFromConfig) {
            const layerName = layerData[0];
            const nodeIndex = layerData[1];
            const tensorIndex = layerData[2];
            generic_utils.assert(layerName in createdLayers);
            const layer = createdLayers[layerName];
            const layerOutputTensors = layer.inboundNodes[nodeIndex].outputTensors;
            outputTensors.push(layerOutputTensors[tensorIndex]);
        }
        return new cls({ inputs: inputTensors, outputs: outputTensors, name });
    }
    /**
     * Determine whether the container is stateful.
     *
     * Porting Note: this is the equivalent of the stateful @property of
     *   the Container class in PyKeras.
     */
    get stateful() {
        // Porting Note: This check is to prevent inadvertent setting of the
        //   _stateful property of the Container instance.
        if (this._stateful) {
            throw new ValueError('Container instance unexpectedly has _stateful = true. The ' +
                'statefulness of a Container is determined by the Layers it ' +
                'contains. Its _stateful property must remain the default false.');
        }
        for (const layer of this.layers) {
            if (layer.stateful) {
                return true;
            }
        }
        return false;
    }
    /**
     * Reset the state of all stateful constituent layers (if any).
     *
     * Examples of stateful layers include RNN layers whose `stateful` property
     * is set as `true`.
     */
    resetStates() {
        tidy(() => {
            this.layers.forEach(layer => {
                // tslint:disable:no-any
                if (layer.stateful) {
                    layer.resetStates();
                }
                // tslint:enable:no-any
            });
        });
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udGFpbmVyLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1sYXllcnMvc3JjL2VuZ2luZS9jb250YWluZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0dBUUc7QUFFSCwrQ0FBK0M7QUFFL0MsT0FBTyxFQUFnRCxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUUxRixPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFDeEMsT0FBTyxFQUFDLG1CQUFtQixFQUFFLFlBQVksRUFBRSxVQUFVLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFJeEUsT0FBTyxFQUFDLFdBQVcsSUFBSSxnQkFBZ0IsRUFBQyxNQUFNLHlCQUF5QixDQUFDO0FBRXhFLE9BQU8sS0FBSyxhQUFhLE1BQU0sd0JBQXdCLENBQUM7QUFDeEQsT0FBTyxFQUFDLG1CQUFtQixFQUFDLE1BQU0sOEJBQThCLENBQUM7QUFDakUsT0FBTyxLQUFLLFdBQVcsTUFBTSxzQkFBc0IsQ0FBQztBQUNwRCxPQUFPLEVBQUMsYUFBYSxFQUFnQixNQUFNLGNBQWMsQ0FBQztBQUMxRCxPQUFPLEVBQUMsT0FBTyxJQUFJLGFBQWEsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUVwRCxPQUFPLEVBQUMsT0FBTyxFQUFFLFFBQVEsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUM3QyxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBQ3pDLE9BQU8sRUFBZ0IsS0FBSyxFQUFFLElBQUksRUFBaUIsTUFBTSxZQUFZLENBQUM7QUFTdEU7Ozs7OztHQU1HO0FBQ0gsTUFBTSxPQUFnQixTQUFVLFNBQVEsS0FBSztJQW9DM0MsWUFBWSxJQUFtQjtRQUM3Qix5Q0FBeUM7UUFDekMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxDQUFDO1FBcEJaLG1CQUFjLEdBQUcsSUFBSSxHQUFHLEVBQVUsQ0FBQztRQXFCakMsSUFBSSxDQUFDLElB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MjAxOCBHb29nbGUgTExDXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlXG4gKiBsaWNlbnNlIHRoYXQgY2FuIGJlIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgb3IgYXRcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vKiBPcmlnaW5hbCBzb3VyY2U6IGtlcmFzL2VuZ2luZS90b3BvbG9neS5weSAqL1xuXG5pbXBvcnQge05hbWVkVGVuc29yTWFwLCBTY2FsYXIsIHNlcmlhbGl6YXRpb24sIFRlbnNvciwgdGlkeX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtnZXRVaWR9IGZyb20gJy4uL2JhY2tlbmQvc3RhdGUnO1xuaW1wb3J0IHtOb3RJbXBsZW1lbnRlZEVycm9yLCBSdW50aW1lRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4uL2Vycm9ycyc7XG5pbXBvcnQge1NoYXBlfSBmcm9tICcuLi9rZXJhc19mb3JtYXQvY29tbW9uJztcbmltcG9ydCB7VGVuc29yS2V5V2l0aEFyZ3NBcnJheX0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L25vZGVfY29uZmlnJztcbmltcG9ydCB7UHlKc29uRGljdH0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L3R5cGVzJztcbmltcG9ydCB7ZGVzZXJpYWxpemUgYXMgZGVzZXJpYWxpemVMYXllcn0gZnJvbSAnLi4vbGF5ZXJzL3NlcmlhbGl6YXRpb24nO1xuaW1wb3J0IHtLd2FyZ3N9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIGdlbmVyaWNfdXRpbHMgZnJvbSAnLi4vdXRpbHMvZ2VuZXJpY191dGlscyc7XG5pbXBvcnQge2NvbnZlcnRUc1RvUHl0aG9uaWN9IGZyb20gJy4uL3V0aWxzL3NlcmlhbGl6YXRpb25fdXRpbHMnO1xuaW1wb3J0ICogYXMgdHlwZXNfdXRpbHMgZnJvbSAnLi4vdXRpbHMvdHlwZXNfdXRpbHMnO1xuaW1wb3J0IHtiYXRjaFNldFZhbHVlLCBMYXllclZhcmlhYmxlfSBmcm9tICcuLi92YXJpYWJsZXMnO1xuaW1wb3J0IHt2ZXJzaW9uIGFzIGxheWVyc1ZlcnNpb259IGZyb20gJy4uL3ZlcnNpb24nO1xuXG5pbXBvcnQge2V4ZWN1dGUsIEZlZWREaWN0fSBmcm9tICcuL2V4ZWN1dG9yJztcbmltcG9ydCB7SW5wdXRMYXllcn0gZnJvbSAnLi9pbnB1dF9sYXllcic7XG5pbXBvcnQge0Rpc3Bvc2VSZXN1bHQsIExheWVyLCBOb2RlLCBTeW1ib2xpY1RlbnNvcn0gZnJvbSAnLi90b3BvbG9neSc7XG5cbi8qKiBDb25zdHJ1Y3RvciBjb25maWcgZm9yIENvbnRhaW5lci4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgQ29udGFpbmVyQXJncyB7XG4gIGlucHV0czogU3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXTtcbiAgb3V0cHV0czogU3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXTtcbiAgbmFtZT86IHN0cmluZztcbn1cblxuLyoqXG4gKiBBIENvbnRhaW5lciBpcyBhIGRpcmVjdGVkIGFjeWNsaWMgZ3JhcGggb2YgbGF5ZXJzLlxuICpcbiAqIEl0IGlzIHRoZSB0b3BvbG9naWNhbCBmb3JtIG9mIGEgXCJtb2RlbFwiLiBBIExheWVyc01vZGVsXG4gKiBpcyBzaW1wbHkgYSBDb250YWluZXIgd2l0aCBhZGRlZCB0cmFpbmluZyByb3V0aW5lcy5cbiAqXG4gKi9cbmV4cG9ydCBhYnN0cmFjdCBjbGFzcyBDb250YWluZXIgZXh0ZW5kcyBMYXllciB7XG4gIGlucHV0czogU3ltYm9saWNUZW5zb3JbXTtcbiAgb3V0cHV0czogU3ltYm9saWNUZW5zb3JbXTtcblxuICBpbnB1dExheWVyczogTGF5ZXJbXTtcbiAgaW5wdXRMYXllcnNOb2RlSW5kaWNlczogbnVtYmVyW107XG4gIGlucHV0TGF5ZXJzVGVuc29ySW5kaWNlczogbnVtYmVyW107XG5cbiAgb3V0cHV0TGF5ZXJzOiBMYXllcltdO1xuICBvdXRwdXRMYXllcnNOb2RlSW5kaWNlczogbnVtYmVyW107XG4gIG91dHB1dExheWVyc1RlbnNvckluZGljZXM6IG51bWJlcltdO1xuXG4gIGxheWVyczogTGF5ZXJbXTtcbiAgbGF5ZXJzQnlEZXB0aDoge1tkZXB0aDogc3RyaW5nXTogTGF5ZXJbXX07XG4gIG5vZGVzQnlEZXB0aDoge1tkZXB0aDogc3RyaW5nXTogTm9kZVtdfTtcblxuICBpbnRlcm5hbENvbnRhaW5lclJlZnM6IENvbnRhaW5lcltdO1xuXG4gIGNvbnRhaW5lck5vZGVzID0gbmV3IFNldDxzdHJpbmc+KCk7XG5cbiAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBBZGQgY2FjaGUgc3VwcG9ydFxuICAvLyBwcml2YXRlIG91dHB1dE1hc2tDYWNoZTogYW55O1xuICAvLyBwcml2YXRlIG91dHB1dFRlbnNvckNhY2hlOiBhbnk7XG4gIC8vIHByaXZhdGUgb3V0cHV0U2hhcGVDYWNoZTogYW55O1xuXG4gIGlucHV0TmFtZXM6IHN0cmluZ1tdO1xuICBvdXRwdXROYW1lczogc3RyaW5nW107XG4gIGZlZWRJbnB1dFNoYXBlczogU2hhcGVbXTtcblxuICBwcm90ZWN0ZWQgaW50ZXJuYWxJbnB1dFNoYXBlczogU2hhcGVbXTtcbiAgcHJvdGVjdGVkIGludGVybmFsT3V0cHV0U2hhcGVzOiBTaGFwZVtdO1xuICAvLyBUT0RPKGNhaXMpOiBNYXliZSAnZmVlZCcgc2hvdWxkIG5vdCBpbiB0aGUgbmFtZXMgb2YgdGhlc2UgdmFyaWFibGVzLFxuICAvLyAgIGR1ZSB0byB0aGUgZmFjdCB0aGF0IG91ciBiYWNrZW5kIGlzIG5vdCBzeW1ib2xpYy5cbiAgcHJvdGVjdGVkIGZlZWRJbnB1dE5hbWVzOiBzdHJpbmdbXTtcbiAgcHJvdGVjdGVkIGZlZWRPdXRwdXROYW1lczogc3RyaW5nW107XG5cbiAgY29uc3RydWN0b3IoYXJnczogQ29udGFpbmVyQXJncykge1xuICAgIC8vIE5vIGFyZ3MgcGFzc2VkIHRvIHN1cGVyJ3MgY29uc3RydWN0b3IuXG4gICAgc3VwZXIoe30pO1xuICAgIHRoaXMubmFtZSA9IGFyZ3MubmFtZTtcbiAgICBpZiAodGhpcy5uYW1lID09IG51bGwpIHtcbiAgICAgIGNvbnN0IHByZWZpeCA9IHRoaXMuZ2V0Q2xhc3NOYW1lKCkudG9Mb3dlckNhc2UoKTtcbiAgICAgIHRoaXMubmFtZSA9IGdldFVpZChwcmVmaXgpO1xuICAgIH1cblxuICAgIHRoaXMuc3VwcG9ydHNNYXNraW5nID0gZmFsc2U7XG4gICAgdGhpcy50cmFpbmFibGVfID0gdHJ1ZTtcblxuICAgIC8vIFRPRE8obWljaGFlbHRlcnJ5KTogSW5pdGlhbGl6ZSBwZXJJbnB1dExvc3Nlcy9VcGRhdGVzIGhlcmUuXG5cbiAgICAvLyBDb250YWluZXItc3BlY2lmaWMgcHJvcGVydGllcy5cbiAgICBpZiAoQXJyYXkuaXNBcnJheShhcmdzLmlucHV0cykpIHtcbiAgICAgIHRoaXMuaW5wdXRzID0gYXJncy5pbnB1dHMuc2xpY2UoKTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5pbnB1dHMgPSBbYXJncy5pbnB1dHNdO1xuICAgIH1cbiAgICBpZiAoQXJyYXkuaXNBcnJheShhcmdzLm91dHB1dHMpKSB7XG4gICAgICB0aGlzLm91dHB1dHMgPSBhcmdzLm91dHB1dHMuc2xpY2UoKTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5vdXRwdXRzID0gW2FyZ3Mub3V0cHV0c107XG4gICAgfVxuXG4gICAgLy8gQ2hlY2sgZm9yIHJlZHVuZGFuY3kgaW4gaW5wdXRzLlxuICAgIGlmIChnZW5lcmljX3V0aWxzLnVuaXF1ZSh0aGlzLmlucHV0cykubGVuZ3RoICE9PSB0aGlzLmlucHV0cy5sZW5ndGgpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICdUaGUgbGlzdCBvZiBpbnB1dHMgcGFzc2VkIHRvIHRoZSBtb2RlbCBpcyAnICtcbiAgICAgICAgICAncmVkdW5kYW50LiBBbGwgaW5wdXRzIHNob3VsZCBvbmx5IGFwcGVhciBvbmNlLiBGb3VuZDogJyArXG4gICAgICAgICAgYCR7dGhpcy5pbnB1dHMubWFwKHggPT4geC5uYW1lKX1gKTtcbiAgICB9XG5cbiAgICAvLyBDaGVjayBmb3IgcmVkdW5kYW5jeSBpbiBvdXRwdXRzLlxuICAgIGlmIChnZW5lcmljX3V0aWxzLnVuaXF1ZSh0aGlzLm91dHB1dHMpLmxlbmd0aCAhPT0gdGhpcy5vdXRwdXRzLmxlbmd0aCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICdUaGUgbGlzdCBvZiBvdXRwdXRzIHBhc3NlZCB0byB0aGUgbW9kZWwgaXMgcmVkdW5kYW50LiAnICtcbiAgICAgICAgICAnQWxsIG91dHB1dHMgc2hvdWxkIG9ubHkgYXBwZWFyIG9uY2UuIEZvdW5kOiAnICtcbiAgICAgICAgICBgJHt0aGlzLm91dHB1dHMubWFwKHggPT4geC5uYW1lKX1gKTtcbiAgICB9XG5cbiAgICAvKlxuICAgICAgTGlzdCBvZiBpbml0aWFsIGxheWVycyAoMSB0byAxIG1hcHBpbmcgd2l0aCB0aGlzLmlucHV0cywgaGVuY2UgdGhlIHNhbWVcbiAgICAgIGxheWVyIG1pZ2h0IGFwcGVhciB0d2ljZSlcbiAgICAqL1xuICAgIHRoaXMuaW5wdXRMYXllcnMgPSBbXTtcbiAgICB0aGlzLmlucHV0TGF5ZXJzTm9kZUluZGljZXMgPSBbXTtcbiAgICB0aGlzLmlucHV0TGF5ZXJzVGVuc29ySW5kaWNlcyA9IFtdO1xuICAgIC8qXG4gICAgICBMaXN0IG9mIGxheWVycyAoMSB0byAxIG1hcHBpbmcgd2l0aCB0aGlzLm91dHB1dHMsIGhlbmNlIHRoZSBzYW1lIGxheWVyXG4gICAgICBtaWdodCBhcHBlYXIgdHdpY2UpXG4gICAgKi9cbiAgICB0aGlzLm91dHB1dExheWVycyA9IFtdO1xuICAgIHRoaXMub3V0cHV0TGF5ZXJzTm9kZUluZGljZXMgPSBbXTtcbiAgICB0aGlzLm91dHB1dExheWVyc1RlbnNvckluZGljZXMgPSBbXTtcbiAgICAvKlxuICAgICAgQWxsIGxheWVycyBpbiBvcmRlciBvZiBob3Jpem9udGFsIGdyYXBoIHRyYXZlcnNhbC4gRW50cmllcyBhcmUgdW5pcXVlLlxuICAgICAgSW5jbHVkZXMgaW5wdXQgYW5kIG91dHB1dCBsYXllcnMuXG4gICAgKi9cbiAgICB0aGlzLmxheWVycyA9IFtdO1xuXG4gICAgLypcbiAgICAgIFJlZmVyZW5jZXMgdG8gY29udGFpbmVyIGxheWVycyB0aGF0IHdlcmUgY29uc3RydWN0ZWQgaW50ZXJuYWxseS4gV2UgbmVlZFxuICAgICAgdGhlc2UgdG8gcHJvcGVybHkgZGlzcG9zZSBvZiB0ZW5zb3JzIGZyb20gbmVzdGVkIGNvbnRhaW5lcnMuXG4gICAgKi9cbiAgICB0aGlzLmludGVybmFsQ29udGFpbmVyUmVmcyA9IFtdO1xuXG4gICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBEZXRlcm1pbmUgaWYgY2FjaGluZyBzdGlsbCBuZWVkZWQgd2l0aCBlYWdlclxuICAgIC8vIGJhY2tlbmQuXG4gICAgLypcbiAgICAgIFRoaXMgaXMgZm9yIHBlcmZvcm1hbmNlIG9wdGltaXphdGlvbiB3aGVuIGNhbGxpbmcgdGhlIENvbnRhaW5lciBvbiBuZXdcbiAgICAgIGlucHV0cy4gRXZlcnkgdGltZSB0aGUgQ29udGFpbmVyIGlzIGNhbGxlZCBvbiBhIHNldCBvbiBpbnB1dCB0ZW5zb3JzLFxuICAgICAgd2UgY29tcHV0ZSB0aGUgb3V0cHV0IHRlbnNvcnMsIG91dHB1dCBtYXNrcyBhbmQgb3V0cHV0IHNoYXBlcyBpbiBvbmUgcGFzcyxcbiAgICAgIHRoZW4gY2FjaGUgdGhlbSBoZXJlLiBXaGVuIG9uZSBvZiB0aGVzZSBvdXRwdXRzIGlzIHF1ZXJpZWQgbGF0ZXIsXG4gICAgICB3ZSByZXRyaWV2ZSBpdCBmcm9tIHRoZXJlIGluc3RlYWQgb2YgcmVjb21wdXRpbmcgaXQuXG4gICAgKi9cbiAgICAvLyB0aGlzLm91dHB1dFRlbnNvckNhY2hlID0ge307XG4gICAgLy8gdGhpcy5vdXRwdXRTaGFwZUNhY2hlID0ge307XG5cbiAgICAvLyBCdWlsZCB0aGlzLm91dHB1dExheWVyczpcbiAgICBmb3IgKGNvbnN0IHggb2YgdGhpcy5vdXRwdXRzKSB7XG4gICAgICBjb25zdCBsYXllciA9IHguc291cmNlTGF5ZXI7XG4gICAgICBjb25zdCBub2RlSW5kZXggPSB4Lm5vZGVJbmRleDtcbiAgICAgIGNvbnN0IHRlbnNvckluZGV4ID0geC50ZW5zb3JJbmRleDtcbiAgICAgIHRoaXMub3V0cHV0TGF5ZXJzLnB1c2gobGF5ZXIpO1xuICAgICAgdGhpcy5vdXRwdXRMYXllcnNOb2RlSW5kaWNlcy5wdXNoKG5vZGVJbmRleCk7XG4gICAgICB0aGlzLm91dHB1dExheWVyc1RlbnNvckluZGljZXMucHVzaCh0ZW5zb3JJbmRleCk7XG4gICAgfVxuXG4gICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBBZGQgb3V0cHV0IG1hc2sgY2FjaGUgY29kZS5cblxuICAgIC8vIEJ1aWxkIHRoaXMuaW5wdXRMYXllcnM6XG4gICAgZm9yIChjb25zdCB4IG9mIHRoaXMuaW5wdXRzKSB7XG4gICAgICBjb25zdCBsYXllciA9IHguc291cmNlTGF5ZXI7XG4gICAgICBjb25zdCBub2RlSW5kZXggPSB4Lm5vZGVJbmRleDtcbiAgICAgIGNvbnN0IHRlbnNvckluZGV4ID0geC50ZW5zb3JJbmRleDtcbiAgICAgIC8qXG4gICAgICAgIEl0J3Mgc3VwcG9zZWQgdG8gYmUgYW4gaW5wdXQgbGF5ZXIsIHNvIG9ubHkgb25lIG5vZGVcbiAgICAgICAgYW5kIG9uZSB0ZW5zb3Igb3V0cHV0LlxuICAgICAgKi9cbiAgICAgIGdlbmVyaWNfdXRpbHMuYXNzZXJ0KG5vZGVJbmRleCA9PT0gMCwgJ2lucHV0IGxheWVyIGhhcyA+MSBub2RlcycpO1xuICAgICAgZ2VuZXJpY191dGlscy5hc3NlcnQodGVuc29ySW5kZXggPT09IDAsICdpbnB1dCBsYXllciBoYXMgPjEgdGVuc29ycycpO1xuICAgICAgdGhpcy5pbnB1dExheWVycy5wdXNoKGxheWVyKTtcbiAgICAgIHRoaXMuaW5wdXRMYXllcnNOb2RlSW5kaWNlcy5wdXNoKG5vZGVJbmRleCk7XG4gICAgICB0aGlzLmlucHV0TGF5ZXJzVGVuc29ySW5kaWNlcy5wdXNoKHRlbnNvckluZGV4KTtcbiAgICB9XG5cbiAgICAvLyBCdWlsZCB0aGlzLmlucHV0TmFtZXMgYW5kIHRoaXMub3V0cHV0TmFtZXMuXG4gICAgdGhpcy5pbnB1dE5hbWVzID0gW107XG4gICAgdGhpcy5vdXRwdXROYW1lcyA9IFtdO1xuICAgIHRoaXMuZmVlZElucHV0U2hhcGVzID0gW107XG4gICAgdGhpcy5mZWVkSW5wdXROYW1lcyA9IFtdO1xuICAgIHRoaXMuZmVlZE91dHB1dE5hbWVzID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLmlucHV0TGF5ZXJzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBsYXllciA9IHRoaXMuaW5wdXRMYXllcnNbaV07XG4gICAgICAvLyBDaGVjayB0aGF0IGxheWVyIGlzIGFuIElucHV0TGF5ZXIuXG4gICAgICBpZiAoIShsYXllciBpbnN0YW5jZW9mIElucHV0TGF5ZXIpKSB7XG4gICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoXG4gICAgICAgICAgICAnSW5wdXQgbGF5ZXJzIHRvIGEgTGF5ZXJzTW9kZWwgbXVzdCBiZSBJbnB1dExheWVyIG9iamVjdHMuICcgK1xuICAgICAgICAgICAgYFJlY2VpdmVkIGlucHV0czogJHthcmdzLmlucHV0c30uIGAgK1xuICAgICAgICAgICAgYElucHV0ICR7aX0gKDAtYmFzZWQpIG9yaWdpbmF0ZXMgYCArXG4gICAgICAgICAgICBgZnJvbSBsYXllciB0eXBlICR7bGF5ZXIuZ2V0Q2xhc3NOYW1lKCl9LmApO1xuICAgICAgfVxuICAgICAgdGhpcy5pbnB1dE5hbWVzLnB1c2gobGF5ZXIubmFtZSk7XG4gICAgICB0aGlzLmZlZWRJbnB1dFNoYXBlcy5wdXNoKGxheWVyLmJhdGNoSW5wdXRTaGFwZSk7XG5cbiAgICAgIHRoaXMuZmVlZElucHV0TmFtZXMucHVzaChsYXllci5uYW1lKTtcbiAgICB9XG4gICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLm91dHB1dExheWVycykge1xuICAgICAgdGhpcy5vdXRwdXROYW1lcy5wdXNoKGxheWVyLm5hbWUpO1xuICAgIH1cblxuICAgIHRoaXMuaW50ZXJuYWxJbnB1dFNoYXBlcyA9IHRoaXMuaW5wdXRzLm1hcCh4ID0+IHguc2hhcGUpO1xuICAgIHRoaXMuaW50ZXJuYWxPdXRwdXRTaGFwZXMgPSB0aGlzLm91dHB1dHMubWFwKHggPT4geC5zaGFwZSk7XG5cbiAgICAvKlxuICAgICAgQ29udGFpbmVyX25vZGVzOiBzZXQgb2Ygbm9kZXMgaW5jbHVkZWQgaW4gdGhlIGdyYXBoIChub3QgYWxsIG5vZGVzXG4gICAgICBpbmNsdWRlZCBpbiB0aGUgbGF5ZXJzIGFyZSByZWxldmFudCB0byB0aGUgY3VycmVudCBncmFwaCkuXG4gICAgKi9cbiAgICAvLyBpZHMgb2YgYWxsIG5vZGVzIHJlbGV2YW50IHRvIHRoZSBDb250YWluZXI6XG4gICAgY29uc3Qgbm9kZXNEZXB0aHM6IHtbbm9kZUlEOiBzdHJpbmddOiBudW1iZXJ9ID0ge307XG4gICAgLy8gVG8gcmVjb3ZlciBub2RlcyBmcm9tIHRoZWlyIElELlxuICAgIGNvbnN0IG5vZGVJRFRvTm9kZToge1tub2RlSUQ6IHN0cmluZ106IE5vZGV9ID0ge307XG4gICAgY29uc3QgbGF5ZXJzRGVwdGhzOiB7W2xheWVySUQ6IHN0cmluZ106IG51bWJlcn0gPSB7fTtcbiAgICAvLyBUbyBsYXllcnMgZnJvbSB0aGVpciBJRC5cbiAgICBjb25zdCBsYXllcklEVG9MYXllcjoge1tsYXllcklEOiBzdHJpbmddOiBMYXllcn0gPSB7fTtcbiAgICBjb25zdCBsYXllckluZGljZXM6IHtbbGF5ZXJJRDogc3RyaW5nXTogbnVtYmVyfSA9IHt9O1xuICAgIGNvbnN0IG5vZGVzSW5EZWNyZWFzaW5nRGVwdGg6IE5vZGVbXSA9IFtdO1xuXG4gICAgLyoqXG4gICAgICogQnVpbGRzIGEgbWFwIG9mIHRoZSBncmFwaCBvZiBsYXllcnMuXG4gICAgICpcbiAgICAgKiBUaGlzIHJlY3Vyc2l2ZWx5IHVwZGF0ZXMgdGhlIG1hcCBgbGF5ZXJJbmRpY2VzYCxcbiAgICAgKiB0aGUgbGlzdCBgbm9kZXNJbkRlY3JlYXNpbmdEZXB0aGAgYW5kIHRoZSBzZXQgYGNvbnRhaW5lck5vZGVzYC5cbiAgICAgKlxuICAgICAqIEBwYXJhbSB0ZW5zb3IgU29tZSB0ZW5zb3IgaW4gYSBncmFwaC5cbiAgICAgKiBAcGFyYW0gZmluaXNoZWROb2RlcyBTZXQgb2Ygbm9kZXMgd2hvc2Ugc3ViZ3JhcGhzIGhhdmUgYmVlbiB0cmF2ZXJzZWRcbiAgICAgKiAgICAgICAgIGNvbXBsZXRlbHkuIFVzZWZ1bCB0byBwcmV2ZW50IGR1cGxpY2F0ZWQgd29yay5cbiAgICAgKiBAcGFyYW0gbm9kZXNJblByb2dyZXNzIFNldCBvZiBub2RlcyB0aGF0IGFyZSBjdXJyZW50bHkgYWN0aXZlIG9uIHRoZVxuICAgICAqICAgICAgICAgcmVjdXJzaW9uIHN0YWNrLiBVc2VmdWwgdG8gZGV0ZWN0IGN5Y2xlcy5cbiAgICAgKiBAcGFyYW0gbGF5ZXIgTGF5ZXIgZnJvbSB3aGljaCBgdGVuc29yYCBjb21lcyBmcm9tLiBJZiBub3QgcHJvdmlkZWQsXG4gICAgICogICB3aWxsIGJlIG9idGFpbmVkIGZyb20gdGVuc29yLnNvdXJjZUxheWVyLlxuICAgICAqIEBwYXJhbSBub2RlSW5kZXggTm9kZSBpbmRleCBmcm9tIHdoaWNoIGB0ZW5zb3JgIGNvbWVzIGZyb20uXG4gICAgICogQHBhcmFtIHRlbnNvckluZGV4IFRlbnNvckluZGV4IGZyb20gd2hpY2ggYHRlbnNvcmAgY29tZXMgZnJvbS5cbiAgICAgKlxuICAgICAqIEBleGNlcHRpb24gUnVudGltZUVycm9yIGlmIGEgY3ljbGUgaXMgZGV0ZWN0ZWQuXG4gICAgICovXG4gICAgY29uc3QgYnVpbGRNYXBPZkdyYXBoID1cbiAgICAgICAgKHRlbnNvcjogU3ltYm9saWNUZW5zb3IsIGZpbmlzaGVkTm9kZXM6IE5vZGVbXSwgbm9kZXNJblByb2dyZXNzOiBOb2RlW10sXG4gICAgICAgICBsYXllcj86IExheWVyLCBub2RlSW5kZXg/OiBudW1iZXIsIHRlbnNvckluZGV4PzogbnVtYmVyKSA9PiB7XG4gICAgICAgICAgaWYgKGxheWVyID09IG51bGwgfHwgbm9kZUluZGV4ID09IG51bGwgfHwgdGVuc29ySW5kZXggPT0gbnVsbCkge1xuICAgICAgICAgICAgbGF5ZXIgPSB0ZW5zb3Iuc291cmNlTGF5ZXI7XG4gICAgICAgICAgICBub2RlSW5kZXggPSB0ZW5zb3Iubm9kZUluZGV4O1xuICAgICAgICAgICAgdGVuc29ySW5kZXggPSB0ZW5zb3IudGVuc29ySW5kZXg7XG4gICAgICAgICAgfVxuICAgICAgICAgIGNvbnN0IG5vZGUgPSBsYXllci5pbmJvdW5kTm9kZXNbbm9kZUluZGV4XTtcblxuICAgICAgICAgIC8vIFByZXZlbnQgY3ljbGVzLlxuICAgICAgICAgIGlmIChub2Rlc0luUHJvZ3Jlc3MuaW5kZXhPZihub2RlKSAhPT0gLTEpIHtcbiAgICAgICAgICAgIHRocm93IG5ldyBSdW50aW1lRXJyb3IoXG4gICAgICAgICAgICAgICAgYFRoZSB0ZW5zb3IgJHt0ZW5zb3IubmFtZX0gYXQgbGF5ZXIgXCIke2xheWVyLm5hbWV9XCIgYCArXG4gICAgICAgICAgICAgICAgJ2lzIHBhcnQgb2YgYSBjeWNsZS4nKTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICAvLyBEb24ndCByZXBlYXQgd29yayBmb3Igc2hhcmVkIHN1YmdyYXBoc1xuICAgICAgICAgIGlmIChmaW5pc2hlZE5vZGVzLmluZGV4T2Yobm9kZSkgIT09IC0xKSB7XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgLy8gVXBkYXRlIGNvbnRhaW5lck5vZGVzLlxuICAgICAgICAgIHRoaXMuY29udGFpbmVyTm9kZXMuYWRkKENvbnRhaW5lci5ub2RlS2V5KGxheWVyLCBub2RlSW5kZXgpKTtcblxuICAgICAgICAgIC8vIFN0b3JlIHRoZSB0cmF2ZXJzYWwgb3JkZXIgZm9yIGxheWVyIHNvcnRpbmcuXG4gICAgICAgICAgaWYgKCEobGF5ZXIuaWQgaW4gbGF5ZXJJbmRpY2VzKSkge1xuICAgICAgICAgICAgbGF5ZXJJbmRpY2VzW2xheWVyLmlkXSA9IE9iamVjdC5rZXlzKGxheWVySW5kaWNlcykubGVuZ3RoO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGlmIChub2Rlc0luUHJvZ3Jlc3MuaW5kZXhPZihub2RlKSA9PT0gLTEpIHtcbiAgICAgICAgICAgIG5vZGVzSW5Qcm9ncmVzcy5wdXNoKG5vZGUpO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIC8vIFByb3BhZ2F0ZSB0byBhbGwgcHJldmlvdXMgdGVuc29ycyBjb25uZWN0ZWQgdG8gdGhpcyBub2RlLlxuICAgICAgICAgIGNvbnN0IG51bUluYm91bmRMYXllcnMgPSBub2RlLmluYm91bmRMYXllcnMubGVuZ3RoO1xuICAgICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbnVtSW5ib3VuZExheWVyczsgaSsrKSB7XG4gICAgICAgICAgICBjb25zdCB4ID0gbm9kZS5pbnB1dFRlbnNvcnNbaV07XG4gICAgICAgICAgICBjb25zdCBsYXllciA9IG5vZGUuaW5ib3VuZExheWVyc1tpXTtcbiAgICAgICAgICAgIGNvbnN0IG5vZGVJbmRleCA9IG5vZGUubm9kZUluZGljZXNbaV07XG4gICAgICAgICAgICBjb25zdCB0ZW5zb3JJbmRleCA9IG5vZGUudGVuc29ySW5kaWNlc1tpXTtcbiAgICAgICAgICAgIGJ1aWxkTWFwT2ZHcmFwaChcbiAgICAgICAgICAgICAgICB4LCBmaW5pc2hlZE5vZGVzLCBub2Rlc0luUHJvZ3Jlc3MsIGxheWVyLCBub2RlSW5kZXgsXG4gICAgICAgICAgICAgICAgdGVuc29ySW5kZXgpO1xuICAgICAgICAgIH1cbiAgICAgICAgICBmaW5pc2hlZE5vZGVzLnB1c2gobm9kZSk7XG4gICAgICAgICAgd2hpbGUgKG5vZGVzSW5Qcm9ncmVzcy5pbmRleE9mKG5vZGUpID49IDApIHtcbiAgICAgICAgICAgIG5vZGVzSW5Qcm9ncmVzcy5zcGxpY2Uobm9kZXNJblByb2dyZXNzLmluZGV4T2Yobm9kZSksIDEpO1xuICAgICAgICAgIH1cbiAgICAgICAgICBub2Rlc0luRGVjcmVhc2luZ0RlcHRoLnB1c2gobm9kZSk7XG4gICAgICAgIH07XG5cbiAgICBjb25zdCBmaW5pc2hlZE5vZGVzOiBOb2RlW10gPSBbXTtcbiAgICBjb25zdCBub2Rlc0luUHJvZ3Jlc3M6IE5vZGVbXSA9IFtdO1xuICAgIGZvciAoY29uc3QgeCBvZiB0aGlzLm91dHB1dHMpIHtcbiAgICAgIGJ1aWxkTWFwT2ZHcmFwaCh4LCBmaW5pc2hlZE5vZGVzLCBub2Rlc0luUHJvZ3Jlc3MpO1xuICAgIH1cblxuICAgIGNvbnN0IHJldmVyc2VkTm9kZXNJbkRlY3JlYXNpbmdEZXB0aCA9XG4gICAgICAgIG5vZGVzSW5EZWNyZWFzaW5nRGVwdGguc2xpY2UoKS5yZXZlcnNlKCk7XG4gICAgZm9yIChjb25zdCBub2RlIG9mIHJldmVyc2VkTm9kZXNJbkRlY3JlYXNpbmdEZXB0aCkge1xuICAgICAgbm9kZUlEVG9Ob2RlW25vZGUuaWRdID0gbm9kZTtcbiAgICAgIC8vIElmIHRoZSBkZXB0aCBpcyBub3Qgc2V0LCB0aGUgbm9kZSBoYXMgbm8gb3V0Ym91bmQgbm9kZXMgKGRlcHRoIDApLlxuICAgICAgaWYgKCEobm9kZS5pZCBpbiBub2Rlc0RlcHRocykpIHtcbiAgICAgICAgbm9kZXNEZXB0aHNbbm9kZS5pZF0gPSAwO1xuICAgICAgfVxuICAgICAgbGV0IGRlcHRoID0gbm9kZXNEZXB0aHNbbm9kZS5pZF07XG5cbiAgICAgIC8vIFVwZGF0ZSB0aGUgZGVwdGggb2YgdGhlIGNvcnJlc3BvbmRpbmcgbGF5ZXJcbiAgICAgIGNvbnN0IHByZXZpb3VzRGVwdGggPVxuICAgICAgICAgIChsYXllcnNEZXB0aHNbbm9kZS5vdXRib3VuZExheWVyLmlkXSA9PSBudWxsID9cbiAgICAgICAgICAgICAgIDAgOlxuICAgICAgICAgICAgICAgbGF5ZXJzRGVwdGhzW25vZGUub3V0Ym91bmRMYXllci5pZF0pO1xuXG4gICAgICAvKlxuICAgICAgICBJZiB3ZSd2ZSBzZWVuIHRoaXMgbGF5ZXIgYmVmb3JlIGF0IGEgaGlnaGVyIGRlcHRoLCB3ZSBzaG91bGQgdXNlIHRoYXRcbiAgICAgICAgZGVwdGggaW5zdGVhZCBvZiB0aGUgbm9kZSBkZXB0aC4gIFRoaXMgaXMgbmVjZXNzYXJ5IGZvciBzaGFyZWQgbGF5ZXJzXG4gICAgICAgIHRoYXQgaGF2ZSBpbnB1dHMgYXQgZGlmZmVyZW50IGRlcHRoIGxldmVscyBpbiB0aGUgZ3JhcGguXG4gICAgICAqL1xuICAgICAgZGVwdGggPSBNYXRoLm1heChkZXB0aCwgcHJldmlvdXNEZXB0aCk7XG4gICAgICBsYXllcnNEZXB0aHNbbm9kZS5vdXRib3VuZExheWVyLmlkXSA9IGRlcHRoO1xuICAgICAgbGF5ZXJJRFRvTGF5ZXJbbm9kZS5vdXRib3VuZExheWVyLmlkXSA9IG5vZGUub3V0Ym91bmRMYXllcjtcbiAgICAgIG5vZGVzRGVwdGhzW25vZGUuaWRdID0gZGVwdGg7XG5cbiAgICAgIC8vIFVwZGF0ZSB0aGUgZGVwdGggb2YgaW5ib3VuZCBub2Rlcy5cbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbm9kZS5pbmJvdW5kTGF5ZXJzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIGNvbnN0IGluYm91bmRMYXllciA9IG5vZGUuaW5ib3VuZExheWVyc1tpXTtcbiAgICAgICAgY29uc3Qgbm9kZUluZGV4ID0gbm9kZS5ub2RlSW5kaWNlc1tpXTtcbiAgICAgICAgY29uc3QgaW5ib3VuZE5vZGUgPSBpbmJvdW5kTGF5ZXIuaW5ib3VuZE5vZGVzW25vZGVJbmRleF07XG4gICAgICAgIGNvbnN0IHByZXZpb3VzRGVwdGggPVxuICAgICAgICAgICAgKG5vZGVzRGVwdGhzW2luYm91bmROb2RlLmlkXSA9PSBudWxsID8gMCA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBub2Rlc0RlcHRoc1tpbmJvdW5kTm9kZS5pZF0pO1xuICAgICAgICBub2Rlc0RlcHRoc1tpbmJvdW5kTm9kZS5pZF0gPSBNYXRoLm1heChkZXB0aCArIDEsIHByZXZpb3VzRGVwdGgpO1xuICAgICAgICBub2RlSURUb05vZGVbaW5ib3VuZE5vZGUuaWRdID0gaW5ib3VuZE5vZGU7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gQnVpbGQgYSBkaWN0IHtkZXB0aDogbGlzdCBvZiBub2RlcyB3aXRoIHRoaXMgZGVwdGh9XG4gICAgY29uc3Qgbm9kZXNCeURlcHRoOiB7W2RlcHRoOiBzdHJpbmddOiBOb2RlW119ID0ge307XG4gICAgZm9yIChjb25zdCBub2RlSUQgaW4gbm9kZXNEZXB0aHMpIHtcbiAgICAgIGNvbnN0IGRlcHRoID0gbm9kZXNEZXB0aHNbbm9kZUlEXTtcbiAgICAgIGlmICghKGRlcHRoIGluIG5vZGVzQnlEZXB0aCkpIHtcbiAgICAgICAgbm9kZXNCeURlcHRoW2RlcHRoXSA9IFtdO1xuICAgICAgfVxuICAgICAgbm9kZXNCeURlcHRoW2RlcHRoXS5wdXNoKG5vZGVJRFRvTm9kZVtub2RlSURdKTtcbiAgICB9XG5cbiAgICAvLyBCdWlsZCBhIGRpY3Qge2RlcHRoOiBsaXN0IG9mIGxheWVycyB3aXRoIHRoaXMgZGVwdGh9XG4gICAgY29uc3QgbGF5ZXJzQnlEZXB0aDoge1tkZXB0aDogc3RyaW5nXTogTGF5ZXJbXX0gPSB7fTtcbiAgICBmb3IgKGNvbnN0IGxheWVySUQgaW4gbGF5ZXJzRGVwdGhzKSB7XG4gICAgICBjb25zdCBkZXB0aCA9IGxheWVyc0RlcHRoc1tsYXllcklEXTtcbiAgICAgIGlmICghKGRlcHRoIGluIGxheWVyc0J5RGVwdGgpKSB7XG4gICAgICAgIGxheWVyc0J5RGVwdGhbZGVwdGhdID0gW107XG4gICAgICB9XG4gICAgICBsYXllcnNCeURlcHRoW2RlcHRoXS5wdXNoKGxheWVySURUb0xheWVyW2xheWVySURdKTtcbiAgICB9XG5cbiAgICAvLyBHZXQgc29ydGVkIGxpc3Qgb2YgbGF5ZXIgZGVwdGhzLlxuICAgIGxldCBkZXB0aEtleXMgPSBPYmplY3Qua2V5cyhsYXllcnNCeURlcHRoKVxuICAgICAgICAgICAgICAgICAgICAgICAgLm1hcCh4ID0+IHBhcnNlSW50KHgsIDEwKSlcbiAgICAgICAgICAgICAgICAgICAgICAgIC5zb3J0KGdlbmVyaWNfdXRpbHMucmV2ZXJzZU51bWJlckNvbXBhcmUpO1xuXG4gICAgLy8gU2V0IHRoaXMubGF5ZXJzIGFuZCB0aGlzLmxheWVyc0J5RGVwdGguXG4gICAgdGhpcy5sYXllcnMgPSBbXTtcbiAgICBmb3IgKGNvbnN0IGRlcHRoIG9mIGRlcHRoS2V5cykge1xuICAgICAgY29uc3QgbGF5ZXJzRm9yRGVwdGggPSBsYXllcnNCeURlcHRoW2RlcHRoXTtcbiAgICAgIC8vIENvbnRhaW5lci5sYXllcnMgbmVlZHMgdG8gaGF2ZSBhIGRldGVybWluaXN0aWMgb3JkZXI6XG4gICAgICAvLyBoZXJlIHdlIG9yZGVyIHRoZW0gYnkgdHJhdmVyc2FsIG9yZGVyLlxuICAgICAgbGF5ZXJzRm9yRGVwdGguc29ydCgoYSwgYikgPT4ge1xuICAgICAgICBjb25zdCBhSW5kZXggPSBsYXllckluZGljZXNbYS5pZF07XG4gICAgICAgIGNvbnN0IGJJbmRleCA9IGxheWVySW5kaWNlc1tiLmlkXTtcbiAgICAgICAgaWYgKGFJbmRleCA8IGJJbmRleCkge1xuICAgICAgICAgIHJldHVybiAtMTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoYUluZGV4ID4gYkluZGV4KSB7XG4gICAgICAgICAgcmV0dXJuIDE7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIDA7XG4gICAgICB9KTtcbiAgICAgIGZvciAoY29uc3QgbGF5ZXIgb2YgbGF5ZXJzRm9yRGVwdGgpIHtcbiAgICAgICAgaWYgKGxheWVyIGluc3RhbmNlb2YgQ29udGFpbmVyKSB7XG4gICAgICAgICAgdGhpcy5pbnRlcm5hbENvbnRhaW5lclJlZnMucHVzaChsYXllcik7XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy5sYXllcnMucHVzaChsYXllcik7XG4gICAgICB9XG4gICAgfVxuICAgIHRoaXMubGF5ZXJzQnlEZXB0aCA9IGxheWVyc0J5RGVwdGg7XG5cbiAgICAvLyBHZXQgc29ydGVkIGxpc3Qgb2Ygbm9kZSBkZXB0aHM7XG4gICAgZGVwdGhLZXlzID0gT2JqZWN0LmtleXMobm9kZXNCeURlcHRoKVxuICAgICAgICAgICAgICAgICAgICAubWFwKHggPT4gcGFyc2VJbnQoeCwgMTApKVxuICAgICAgICAgICAgICAgICAgICAuc29ydChnZW5lcmljX3V0aWxzLnJldmVyc2VOdW1iZXJDb21wYXJlKTtcblxuICAgIC8vIENoZWNrIHRoYXQgYWxsIHRlbnNvcnMgcmVxdWlyZWQgYXJlIGNvbXB1dGFibGUuXG4gICAgLy8gY29tcHV0YWJsZV90ZW5zb3JzOiBhbGwgdGVuc29ycyBpbiB0aGUgZ3JhcGhcbiAgICAvLyB0aGF0IGNhbiBiZSBjb21wdXRlZCBmcm9tIHRoZSBpbnB1dHMgcHJvdmlkZWQuXG4gICAgY29uc3QgY29tcHV0YWJsZVRlbnNvcnMgPSB0aGlzLmlucHV0cy5zbGljZSgpO1xuXG4gICAgLy8gVG8gcHJvdmlkZSBhIGJldHRlciBlcnJvciBtc2cuXG4gICAgY29uc3QgbGF5ZXJzV2l0aENvbXBsZXRlSW5wdXQ6IHN0cmluZ1tdID0gW107XG4gICAgZm9yIChjb25zdCBkZXB0aCBvZiBkZXB0aEtleXMpIHtcbiAgICAgIGZvciAoY29uc3Qgbm9kZSBvZiBub2Rlc0J5RGVwdGhbZGVwdGhdKSB7XG4gICAgICAgIGNvbnN0IGxheWVyID0gbm9kZS5vdXRib3VuZExheWVyO1xuICAgICAgICBpZiAobGF5ZXIgIT0gbnVsbCkge1xuICAgICAgICAgIGZvciAoY29uc3QgeCBvZiBub2RlLmlucHV0VGVuc29ycykge1xuICAgICAgICAgICAgaWYgKGNvbXB1dGFibGVUZW5zb3JzLmluZGV4T2YoeCkgPT09IC0xKSB7XG4gICAgICAgICAgICAgIHRocm93IG5ldyBSdW50aW1lRXJyb3IoXG4gICAgICAgICAgICAgICAgICBgR3JhcGggZGlzY29ubmVjdGVkOiBjYW5ub3Qgb2J0YWluIHZhbHVlIGZvciB0ZW5zb3IgJHt4fWAgK1xuICAgICAgICAgICAgICAgICAgYCBhdCBsYXllciBcIiR7bGF5ZXIubmFtZX1cIi4gYCArXG4gICAgICAgICAgICAgICAgICAnVGhlIGZvbGxvd2luZyBwcmV2aW91cyBsYXllcnMgd2VyZSBhY2Nlc3NlZCB3aXRob3V0ICcgK1xuICAgICAgICAgICAgICAgICAgYGlzc3VlOiAke2xheWVyc1dpdGhDb21wbGV0ZUlucHV0fWApO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgICBmb3IgKGNvbnN0IHggb2Ygbm9kZS5vdXRwdXRUZW5zb3JzKSB7XG4gICAgICAgICAgICBjb21wdXRhYmxlVGVuc29ycy5wdXNoKHgpO1xuICAgICAgICAgIH1cbiAgICAgICAgICBsYXllcnNXaXRoQ29tcGxldGVJbnB1dC5wdXNoKGxheWVyLm5hbWUpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gU2V0IHRoaXMuY29udGFpbmVyTm9kZXMgYW5kIHRoaXMubm9kZXNCeURlcHRoLlxuICAgIHRoaXMubm9kZXNCeURlcHRoID0gbm9kZXNCeURlcHRoO1xuXG4gICAgLy8gRW5zdXJlIG5hbWUgdW5pY2l0eSwgd2hpY2ggd2lsbCBiZSBjcnVjaWFsIGZvciBzZXJpYWxpemF0aW9uXG4gICAgLy8gKHNpbmNlIHNlcmlhbGl6ZWQgbm9kZXMgcmVmZXIgdG8gbGF5ZXJzIGJ5IHRoZWlyIG5hbWUpLlxuICAgIGNvbnN0IGFsbE5hbWVzID0gdGhpcy5sYXllcnMubWFwKHggPT4geC5uYW1lKTtcbiAgICBmb3IgKGNvbnN0IG5hbWUgb2YgYWxsTmFtZXMpIHtcbiAgICAgIGNvbnN0IG51bU9jY3VycmVuY2VzID0gYWxsTmFtZXMuZmlsdGVyKHggPT4geCA9PT0gbmFtZSkubGVuZ3RoO1xuICAgICAgaWYgKG51bU9jY3VycmVuY2VzICE9PSAxKSB7XG4gICAgICAgIHRocm93IG5ldyBSdW50aW1lRXJyb3IoXG4gICAgICAgICAgICBgVGhlIG5hbWUgXCIke25hbWV9XCIgaXMgdXNlZCAke251bU9jY3VycmVuY2VzfSB0aW1lcyBgICtcbiAgICAgICAgICAgICdpbiB0aGUgbW9kZWwuIEFsbCBsYXllciBuYW1lcyBzaG91bGQgYmUgdW5pcXVlLiBMYXllciBuYW1lczogJyArXG4gICAgICAgICAgICBKU09OLnN0cmluZ2lmeShhbGxOYW1lcykpO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIExheWVyIHBhcmFtZXRlcnMuXG4gICAgLy8gVGhlIG5ldyBjb250YWluZXIgc3RhcnRzIHdpdGggYSBzaW5nbGUgaW5ib3VuZCBub2RlXG4gICAgLy8gZm9yIGl0cyBpbnB1dHMsIGFuZCBubyBvdXRib3VuZCBub2Rlcy5cbiAgICAvLyBXaWxsIGJlIGFwcGVuZGVkIHRvIGJ5IGZ1dHVyZSBjYWxscyB0byBhcHBseSgpLlxuICAgIHRoaXMub3V0Ym91bmROb2RlcyA9IFtdO1xuICAgIC8vIFdpbGwgYmUgYXBwZW5kZWQgdG8gYmVsb3csIGFuZCBieSBmdXR1cmUgY2FsbHMgdG8gYXBwbHkoKS5cbiAgICB0aGlzLmluYm91bmROb2RlcyA9IFtdO1xuXG4gICAgLy8gQ3JlYXRlIHRoZSBub2RlIGxpbmtpbmcgaW50ZXJuYWwgaW5wdXRzIHRvIGludGVybmFsIG91dHB1dHMuXG4gICAgLy8gKFRoaXMgY2FsbCBoYXMgc2lkZSBlZmZlY3RzLilcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tdW51c2VkLWV4cHJlc3Npb25cbiAgICBuZXcgTm9kZSh7XG4gICAgICBvdXRib3VuZExheWVyOiB0aGlzLFxuICAgICAgaW5ib3VuZExheWVyczogW10sXG4gICAgICBub2RlSW5kaWNlczogW10sXG4gICAgICB0ZW5zb3JJbmRpY2VzOiBbXSxcbiAgICAgIGlucHV0VGVuc29yczogdGhpcy5pbnB1dHMsXG4gICAgICBvdXRwdXRUZW5zb3JzOiB0aGlzLm91dHB1dHMsXG4gICAgICBpbnB1dE1hc2tzOiB0aGlzLmlucHV0cy5tYXAoeCA9PiBudWxsKSxcbiAgICAgIG91dHB1dE1hc2tzOiB0aGlzLm91dHB1dHMubWFwKHggPT4gbnVsbCksXG4gICAgICBpbnB1dFNoYXBlczogdGhpcy5pbnB1dHMubWFwKHggPT4geC5zaGFwZSksXG4gICAgICBvdXRwdXRTaGFwZXM6IHRoaXMub3V0cHV0cy5tYXAoeCA9PiB4LnNoYXBlKVxuICAgIH0pO1xuICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuICAgIHRoaXMuX3JlZkNvdW50ID0gMTsgIC8vIFRoZSByZWYgY291bnQgb2YgYSBjb250YWluZXIgYWx3YXlzIHN0YXJ0IGF0IDEuXG4gIH1cblxuICBwcm90ZWN0ZWQgb3ZlcnJpZGUgYXNzZXJ0Tm90RGlzcG9zZWQoKSB7XG4gICAgaWYgKHRoaXMuX3JlZkNvdW50ID09PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYENvbnRhaW5lciAnJHt0aGlzLm5hbWV9JyBpcyBhbHJlYWR5IGRpc3Bvc2VkLmApO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBBdHRlbXB0IHRvIGRpc3Bvc2UgYSBMYXllcnNNb2RlbCdzIHdlaWdodHMuXG4gICAqXG4gICAqIFRoaXMgbWV0aG9kIGRlY3JlYXNlIHRoZSByZWZlcmVuY2UgY291bnQgb2YgdGhlIExheWVyc01vZGVsIG9iamVjdCBieSAxLlxuICAgKlxuICAgKiBBIExheWVyc01vZGVsIGlzIHJlZmVyZW5jZS1jb3VudGVkLiBJdHMgcmVmZXJlbmNlIGNvdW50IGlzIGluY3JlbWVudGVkIGJ5IDFcbiAgICogd2hlbiBpdCBpcyBmaXJzdCBjb25zdHJ1Y3RlZCBhbmQgd2hlbiBpdCBpcyB1c2VkIGFzIGEgTGF5ZXIgb2YgYW5vdGhlclxuICAgKiBMYXllcnNNb2RlbC5cbiAgICpcbiAgICogSWYgdGhlIHJlZmVyZW5jZSBjb3VudCBvZiBhIExheWVyc01vZGVsIGJlY29tZXMgMCwgdGhlIGBkaXNwb3NlYCBtZXRob2Qgb2ZcbiAgICogYWxsIGl0cyBjb25zdGl0dWVudCBgTGF5ZXJgcyB3aWxsIGJlIGNhbGxlZC5cbiAgICpcbiAgICogTm90ZTogSWYgdGhlIHJlZmVyZW5jZSBjb3VudCBpcyBncmVhdGVyIHRoYW4gMCBhZnRlciB0aGUgZGVjcmVtZW50LCB0aGVcbiAgICogYGRpc3Bvc2VgIG1ldGhvZCBvZiBpdHMgY29uc3RpdHVlbnQgYExheWVyYHMgd2lsbCAqbm90KiBiZSBjYWxsZWQuXG4gICAqXG4gICAqIEFmdGVyIGEgTGF5ZXJzTW9kZWwgaXMgZGlzcG9zZWQsIGl0IGNhbm5vdCBiZSB1c2VkIGluIGNhbGxzIHN1Y2ggYXNcbiAgICogJ3ByZWRpY3RgLCBgZXZhbHVhdGVgIG9yIGBmaXRgIGFueW1vcmUuXG4gICAqXG4gICAqIEByZXR1cm5zIEEgRGlzcG9zZVJlc3VsdCBPYmplY3Qgd2l0aCB0aGUgZm9sbG93aW5nIGZpZWxkczpcbiAgICogICAtIHJlZkNvdW50QWZ0ZXJEaXNwb3NlOiBUaGUgcmVmZXJlbmNlIGNvdW50IG9mIHRoZSBMYXllcnNNb2RlbCBhZnRlciB0aGlzXG4gICAqICAgICBgZGlzcG9zZSgpYCBjYWxsLlxuICAgKiAgIC0gbnVtRGlzcG9zZWRWYXJpYWJsZXM6IE51bWJlciBvZiBgdGYuVmFyaWFibGVgcyAoaS5lLiwgd2VpZ2h0cykgZGlzcG9zZWRcbiAgICogICAgIGR1cmluZyB0aGlzIGBkaXNwb3NlKClgIGNhbGwuXG4gICAqIEB0aHJvd3Mge0Vycm9yfSBJZiB0aGUgbGF5ZXIgaXMgbm90IGJ1aWx0IHlldCwgb3IgaWYgdGhlIExheWVyc01vZGVsIGhhc1xuICAgKiAgIGFscmVhZHkgYmVlbiBkaXNwb3NlZC5cbiAgICovXG4gIG92ZXJyaWRlIGRpc3Bvc2UoKTogRGlzcG9zZVJlc3VsdCB7XG4gICAgdGhpcy5hc3NlcnROb3REaXNwb3NlZCgpO1xuICAgIGNvbnN0IHJlc3VsdDpcbiAgICAgICAgRGlzcG9zZVJlc3VsdCA9IHtyZWZDb3VudEFmdGVyRGlzcG9zZTogbnVsbCwgbnVtRGlzcG9zZWRWYXJpYWJsZXM6IDB9O1xuICAgIGlmICgtLXRoaXMuX3JlZkNvdW50ID09PSAwKSB7XG4gICAgICBmb3IgKGNvbnN0IGxheWVyIG9mIHRoaXMubGF5ZXJzKSB7XG4gICAgICAgIHJlc3VsdC5udW1EaXNwb3NlZFZhcmlhYmxlcyArPSBsYXllci5kaXNwb3NlKCkubnVtRGlzcG9zZWRWYXJpYWJsZXM7XG4gICAgICB9XG5cbiAgICAgIC8vIENhbGwgZGlzcG9zZSBvbiBlYWNoIGludGVybmFsbHkgY3JlYXRlZCBjb250YWluZXIgbGF5ZXIgYWdhaW4gdG8gZW5zdXJlXG4gICAgICAvLyB0aGVpciByZWZDb3VudHMgaGl0IHplcm8gYW5kIHRoZWlyIHRlbnNvcnMgYXJlIHN1YnNlcXVlbnRseSBkZWxldGVkLlxuICAgICAgZm9yIChjb25zdCBjb250YWluZXIgb2YgdGhpcy5pbnRlcm5hbENvbnRhaW5lclJlZnMpIHtcbiAgICAgICAgcmVzdWx0Lm51bURpc3Bvc2VkVmFyaWFibGVzICs9IGNvbnRhaW5lci5kaXNwb3NlKCkubnVtRGlzcG9zZWRWYXJpYWJsZXM7XG4gICAgICB9XG4gICAgfVxuICAgIHJlc3VsdC5yZWZDb3VudEFmdGVyRGlzcG9zZSA9IHRoaXMuX3JlZkNvdW50O1xuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBvdmVycmlkZSBnZXQgdHJhaW5hYmxlKCkge1xuICAgIHJldHVybiB0aGlzLnRyYWluYWJsZV87XG4gIH1cblxuICBvdmVycmlkZSBzZXQgdHJhaW5hYmxlKHRyYWluYWJsZTogYm9vbGVhbikge1xuICAgIHRoaXMubGF5ZXJzLmZvckVhY2gobGF5ZXIgPT4ge1xuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgKChsYXllciBhcyBhbnkpLl90cmFpbmFibGVXZWlnaHRzIGFzIExheWVyVmFyaWFibGVbXSlcbiAgICAgICAgICAuZm9yRWFjaCh3ID0+IHcudHJhaW5hYmxlID0gdHJhaW5hYmxlKTtcbiAgICB9KTtcbiAgICB0aGlzLnRyYWluYWJsZV8gPSB0cmFpbmFibGU7XG4gIH1cblxuICBvdmVycmlkZSBnZXQgdHJhaW5hYmxlV2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIC8vIFBvcnRpbmcgTm90ZTogVGhpcyBjaGVjayBiZWxvdyBpcyB0byBwcmV2ZW50IGVycm9ycyB3aGVyZSB0aGVcbiAgICAvLyAgIF90cmFpbmFibGVXZWlnaHRzIGluaGVyaXRlZCBmcm9tIHRoZSBwYXJlbnQgY2xhc3MgKExheWVyKSBnZXRzXG4gICAgLy8gICBpbmFkdmVydGVudGx5IHVzZWQuXG4gICAgaWYgKHRoaXMuX3RyYWluYWJsZVdlaWdodHMubGVuZ3RoID4gMCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ0NvbnRhaW5lciBpbnN0YW5jZSB1bmV4cGVjdGVkbHkgY29udGFpbnMgX3RyYWluYWJsZVdlaWdodHMuJyArXG4gICAgICAgICAgJ1RoZSB0cmFpbmFibGUgd2VpZ2h0cyBvZiBhIENvbnRhaW5lciBhcmUgYSB1bmlvbiBvZiB0aGUgJyArXG4gICAgICAgICAgJ3RyYWluYWJsZSB3ZWlnaHRzIG9mIGl0cyBjb25zaXR1ZW50IExheWVycy4gSXRzIG93biAnICtcbiAgICAgICAgICAnX3RyYWluYWJsZVdlaWdodHMgbXVzdCByZW1haW4gYW4gZW1wdHkgQXJyYXkuJyk7XG4gICAgfVxuXG4gICAgaWYgKCF0aGlzLnRyYWluYWJsZSkge1xuICAgICAgcmV0dXJuIFtdO1xuICAgIH1cbiAgICBsZXQgd2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdID0gW107XG4gICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLmxheWVycykge1xuICAgICAgd2VpZ2h0cyA9IHdlaWdodHMuY29uY2F0KGxheWVyLnRyYWluYWJsZVdlaWdodHMpO1xuICAgIH1cbiAgICByZXR1cm4gd2VpZ2h0cztcbiAgfVxuXG4gIG92ZXJyaWRlIGdldCBub25UcmFpbmFibGVXZWlnaHRzKCk6IExheWVyVmFyaWFibGVbXSB7XG4gICAgY29uc3Qgd2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdID0gW107XG4gICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLmxheWVycykge1xuICAgICAgd2VpZ2h0cy5wdXNoKC4uLmxheWVyLm5vblRyYWluYWJsZVdlaWdodHMpO1xuICAgIH1cbiAgICBpZiAoIXRoaXMudHJhaW5hYmxlKSB7XG4gICAgICBjb25zdCB0cmFpbmFibGVXZWlnaHRzOiBMYXllclZhcmlhYmxlW10gPSBbXTtcbiAgICAgIGZvciAoY29uc3QgbGF5ZXIgb2YgdGhpcy5sYXllcnMpIHtcbiAgICAgICAgdHJhaW5hYmxlV2VpZ2h0cy5wdXNoKC4uLmxheWVyLnRyYWluYWJsZVdlaWdodHMpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHRyYWluYWJsZVdlaWdodHMuY29uY2F0KHdlaWdodHMpO1xuICAgIH1cbiAgICByZXR1cm4gd2VpZ2h0cztcbiAgfVxuXG4gIG92ZXJyaWRlIGdldCB3ZWlnaHRzKCk6IExheWVyVmFyaWFibGVbXSB7XG4gICAgcmV0dXJuIHRoaXMudHJhaW5hYmxlV2VpZ2h0cy5jb25jYXQodGhpcy5ub25UcmFpbmFibGVXZWlnaHRzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBMb2FkcyBhbGwgbGF5ZXIgd2VpZ2h0cyBmcm9tIGEgSlNPTiBvYmplY3QuXG4gICAqXG4gICAqIFBvcnRpbmcgTm90ZTogSERGNSB3ZWlnaHQgZmlsZXMgY2Fubm90IGJlIGRpcmVjdGx5IGxvYWRlZCBpbiBKYXZhU2NyaXB0IC9cbiAgICogICBUeXBlU2NyaXB0LiBUaGUgdXRpbGl0eSBzY3JpcHQgYXQgYHNjcmlwdHMvcHlrZXJhcy5weWAgb2ZmZXJzIG1lYW5zXG4gICAqICAgdG8gY29udmVydCB0aGVtIGludG8gSlNPTiBzdHJpbmdzIGNvbXBhdGlibGUgd2l0aCB0aGlzIG1ldGhvZC5cbiAgICogUG9ydGluZyBOb3RlOiBUZW5zb3JGbG93LmpzIExheWVycyBzdXBwb3J0cyBvbmx5IGxvYWRpbmcgYnkgbmFtZSBjdXJyZW50bHkuXG4gICAqXG4gICAqIEBwYXJhbSB3ZWlnaHRzIEEgSlNPTiBtYXBwaW5nIHdlaWdodCBuYW1lcyB0byB3ZWlnaHQgdmFsdWVzIGFzIG5lc3RlZFxuICAgKiAgIGFycmF5cyBvZiBudW1iZXJzLCBvciBhIGBOYW1lZFRlbnNvck1hcGAsIGkuZS4sIGEgSlNPTiBtYXBwaW5nIHdlaWdodFxuICAgKiAgIG5hbWVzIHRvIGB0Zi5UZW5zb3JgIG9iamVjdHMuXG4gICAqIEBwYXJhbSBzdHJpY3QgUmVxdWlyZSB0aGF0IHRoZSBwcm92aWRlZCB3ZWlnaHRzIGV4YWN0bHkgbWF0Y2ggdGhvc2VcbiAgICogICByZXF1aXJlZCBieSB0aGUgY29udGFpbmVyLiAgRGVmYXVsdDogYHRydWVgLiAgUGFzc2luZyBgZmFsc2VgIG1lYW5zIHRoYXRcbiAgICogICBleHRyYSB3ZWlnaHRzIGFuZCBtaXNzaW5nIHdlaWdodHMgd2lsbCBiZSBzaWxlbnRseSBpZ25vcmVkLlxuICAgKi9cbiAgbG9hZFdlaWdodHMod2VpZ2h0czogTmFtZWRUZW5zb3JNYXAsIHN0cmljdCA9IHRydWUpIHtcbiAgICBjb25zdCBuYW1lVG9XZWlnaHQ6IHtbbmFtZTogc3RyaW5nXTogTGF5ZXJWYXJpYWJsZX0gPSB7fTtcbiAgICBsZXQgdG90YWxXZWlnaHRzQ291bnQgPSAwO1xuICAgIGZvciAoY29uc3QgbGF5ZXIgb2YgdGhpcy5sYXllcnMpIHtcbiAgICAgIGZvciAoY29uc3Qgd2VpZ2h0IG9mIGxheWVyLndlaWdodHMpIHtcbiAgICAgICAgaWYgKG5hbWVUb1dlaWdodFt3ZWlnaHQub3JpZ2luYWxOYW1lXSAhPSBudWxsKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoYER1cGxpY2F0ZSB3ZWlnaHQgbmFtZTogJHt3ZWlnaHQub3JpZ2luYWxOYW1lfWApO1xuICAgICAgICB9XG4gICAgICAgIG5hbWVUb1dlaWdodFt3ZWlnaHQub3JpZ2luYWxOYW1lXSA9IHdlaWdodDtcbiAgICAgICAgdG90YWxXZWlnaHRzQ291bnQrKztcbiAgICAgIH1cbiAgICB9XG5cbiAgICBjb25zdCB3ZWlnaHRWYWx1ZVR1cGxlczogQXJyYXk8W0xheWVyVmFyaWFibGUsIFRlbnNvcl0+ID0gW107XG4gICAgZm9yIChjb25zdCBuYW1lIGluIHdlaWdodHMpIHtcbiAgICAgIC8vIFRGIDIuMi4wIGFkZGVkIGNlbGwgbmFtZSB0byB0aGUgd2VpZ2h0IG5hbWUgaW4gdGhlIGZvcm1hdCBvZlxuICAgICAgLy8gbGF5ZXJfbmFtZS9jZWxsX25hbWUvd2VpZ2h0X25hbWUsIHdlIG5lZWQgdG8gcmVtb3ZlXG4gICAgICAvLyB0aGUgaW5uZXIgY2VsbCBuYW1lLlxuICAgICAgbGV0IHZhbGlkYXRlZE5hbWUgPSBuYW1lO1xuICAgICAgaWYgKG5hbWVUb1dlaWdodFtuYW1lXSA9PSBudWxsKSB7XG4gICAgICAgIGNvbnN0IHRva2VucyA9IG5hbWUuc3BsaXQoJy8nKTtcbiAgICAgICAgY29uc3Qgc2hvcnRlbk5hbWVBcnJheSA9XG4gICAgICAgICAgICB0b2tlbnMuc2xpY2UoMCwgLTIpLmNvbmNhdChbdG9rZW5zW3Rva2Vucy5sZW5ndGggLSAxXV0pO1xuICAgICAgICB2YWxpZGF0ZWROYW1lID0gc2hvcnRlbk5hbWVBcnJheS5qb2luKCcvJyk7XG4gICAgICB9XG4gICAgICBpZiAobmFtZVRvV2VpZ2h0W3ZhbGlkYXRlZE5hbWVdICE9IG51bGwpIHtcbiAgICAgICAgd2VpZ2h0VmFsdWVUdXBsZXMucHVzaChbbmFtZVRvV2VpZ2h0W3ZhbGlkYXRlZE5hbWVdLCB3ZWlnaHRzW25hbWVdXSk7XG4gICAgICB9IGVsc2UgaWYgKHN0cmljdCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBQcm92aWRlZCB3ZWlnaHQgZGF0YSBoYXMgbm8gdGFyZ2V0IHZhcmlhYmxlOiAke25hbWV9YCk7XG4gICAgICB9XG4gICAgICBkZWxldGUgbmFtZVRvV2VpZ2h0W3ZhbGlkYXRlZE5hbWVdO1xuICAgIH1cblxuICAgIGlmIChzdHJpY3QpIHtcbiAgICAgIC8vIENoZWNrIHRoYXQgYWxsIHdlaWdodHMgYXJlIHNldC5cbiAgICAgIGNvbnN0IHVuc2V0TmFtZXM6IHN0cmluZ1tdID0gW107XG4gICAgICBmb3IgKGNvbnN0IG5hbWUgaW4gbmFtZVRvV2VpZ2h0KSB7XG4gICAgICAgIHVuc2V0TmFtZXMucHVzaChuYW1lKTtcbiAgICAgIH1cbiAgICAgIGlmICh1bnNldE5hbWVzLmxlbmd0aCA+IDApIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgJHt1bnNldE5hbWVzLmxlbmd0aH0gb2YgJHtcbiAgICAgICAgICAgICAgICB0b3RhbFdlaWdodHNDb3VudH0gd2VpZ2h0cyBhcmUgbm90IHNldDogYCArXG4gICAgICAgICAgICBgJHt1bnNldE5hbWVzfWApO1xuICAgICAgfVxuICAgIH1cblxuICAgIGJhdGNoU2V0VmFsdWUod2VpZ2h0VmFsdWVUdXBsZXMpO1xuICB9XG5cbiAgLyoqXG4gICAqIFV0aWwgc2hhcmVkIGJldHdlZW4gZGlmZmVyZW50IHNlcmlhbGl6YXRpb24gbWV0aG9kcy5cbiAgICogQHJldHVybnMgTGF5ZXJzTW9kZWwgY29uZmlnIHdpdGggS2VyYXMgdmVyc2lvbiBpbmZvcm1hdGlvbiBhZGRlZC5cbiAgICovXG4gIHByb3RlY3RlZCB1cGRhdGVkQ29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgdGhlQ29uZmlnID0gdGhpcy5nZXRDb25maWcoKTtcbiAgICBjb25zdCBtb2RlbENvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge307XG4gICAgbW9kZWxDb25maWdbJ2NsYXNzTmFtZSddID0gdGhpcy5nZXRDbGFzc05hbWUoKTtcbiAgICBtb2RlbENvbmZpZ1snY29uZmlnJ10gPSB0aGVDb25maWc7XG4gICAgbW9kZWxDb25maWdbJ2tlcmFzVmVyc2lvbiddID0gYHRmanMtbGF5ZXJzICR7bGF5ZXJzVmVyc2lvbn1gO1xuICAgIC8vIFRPRE8obmllbHNlbmUpOiBSZXBsYWNlIHNvbWV0aGluZyBsaWtlIEsuYmFja2VuZCgpIG9uY2VcbiAgICAvLyBwb3NzaWJsZS5cbiAgICBtb2RlbENvbmZpZ1snYmFja2VuZCddID0gJ1RlbnNvckZsb3cuanMnO1xuICAgIHJldHVybiBtb2RlbENvbmZpZztcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXR1cm5zIGEgSlNPTiBzdHJpbmcgY29udGFpbmluZyB0aGUgbmV0d29yayBjb25maWd1cmF0aW9uLlxuICAgKlxuICAgKiBUbyBsb2FkIGEgbmV0d29yayBmcm9tIGEgSlNPTiBzYXZlIGZpbGUsIHVzZVxuICAgKiBtb2RlbHMubW9kZWxGcm9tSlNPTihqc29uU3RyaW5nKTtcbiAgICogQHBhcmFtIGV4dHJhSnNvbkFyZ3MgVW51c2VkIGluIHRmanMtbGF5ZXJzLCBtYWludGFpbmVkIGZvciBQeUtlcmFzXG4gICAqIEBwYXJhbSByZXR1cm5TdHJpbmcgV2hldGhlciB0aGUgcmV0dXJuIHZhbHVlIHNob3VsZCBiZSBzdHJpbmdpZmllZFxuICAgKiAgICAoZGVmYXVsdDogYHRydWVgKS5cbiAgICogQHJldHVybnMgYSBKU09OIHN0cmluZyBpZiBgcmV0dXJuU3RyaW5nYCAoZGVmYXVsdCksIG9yIGEgSlNPTiBvYmplY3QgaWZcbiAgICogICBgIXJldHVyblN0cmluZ2AuXG4gICAqL1xuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gIHRvSlNPTih1bnVzZWQ/OiBhbnksIHJldHVyblN0cmluZyA9IHRydWUpOiBzdHJpbmd8UHlKc29uRGljdCB7XG4gICAgY29uc3QgbW9kZWxDb25maWcgPSBjb252ZXJ0VHNUb1B5dGhvbmljKHRoaXMudXBkYXRlZENvbmZpZygpKSBhcyBQeUpzb25EaWN0O1xuICAgIHJldHVybiByZXR1cm5TdHJpbmcgPyBKU09OLnN0cmluZ2lmeShtb2RlbENvbmZpZykgOiBtb2RlbENvbmZpZztcbiAgfVxuXG4gIC8qKlxuICAgKiBDYWxsIHRoZSBtb2RlbCBvbiBuZXcgaW5wdXRzLlxuICAgKlxuICAgKiBJbiB0aGlzIGNhc2UgYGNhbGxgIGp1c3QgcmVhcHBsaWVzIGFsbCBvcHMgaW4gdGhlIGdyYXBoIHRvIHRoZSBuZXcgaW5wdXRzXG4gICAqIChlLmcuIGJ1aWxkIGEgbmV3IGNvbXB1dGF0aW9uYWwgZ3JhcGggZnJvbSB0aGUgcHJvdmlkZWQgaW5wdXRzKS5cbiAgICpcbiAgICogQHBhcmFtIGlucHV0cyBBIHRlbnNvciBvciBsaXN0IG9mIHRlbnNvcnMuXG4gICAqIEBwYXJhbSBtYXNrIEEgbWFzayBvciBsaXN0IG9mIG1hc2tzLiBBIG1hc2sgY2FuIGJlIGVpdGhlciBhIHRlbnNvciBvciBudWxsXG4gICAqICAgKG5vIG1hc2spLlxuICAgKlxuICAgKiBAcmV0dXJuIEEgdGVuc29yIGlmIHRoZXJlIGlzIGEgc2luZ2xlIG91dHB1dCwgb3IgYSBsaXN0IG9mIHRlbnNvcnMgaWYgdGhlcmVcbiAgICogICBhcmUgbW9yZSB0aGFuIG9uZSBvdXRwdXRzLlxuICAgKi9cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGlucHV0cyA9IGdlbmVyaWNfdXRpbHMudG9MaXN0KGlucHV0cyk7XG4gICAgICBjb25zdCBmZWVkRGljdCA9IG5ldyBGZWVkRGljdCgpO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLmlucHV0cy5sZW5ndGg7ICsraSkge1xuICAgICAgICBmZWVkRGljdC5hZGQodGhpcy5pbnB1dHNbaV0sIGlucHV0c1tpXSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gZXhlY3V0ZSh0aGlzLm91dHB1dHMsIGZlZWREaWN0LCBrd2FyZ3MpIGFzIFRlbnNvciB8IFRlbnNvcltdO1xuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGFuIG91dHB1dCBtYXNrIHRlbnNvci5cbiAgICpcbiAgICogQHBhcmFtIGlucHV0cyBUZW5zb3Igb3IgbGlzdCBvZiB0ZW5zb3JzLlxuICAgKiBAcGFyYW0gbWFzayBUZW5zb3Igb3IgbGlzdCBvZiB0ZW5zb3JzLlxuICAgKlxuICAgKiBAcmV0dXJuIG51bGwgb3IgYSB0ZW5zb3IgKG9yIGxpc3Qgb2YgdGVuc29ycywgb25lIHBlciBvdXRwdXQgdGVuc29yIG9mIHRoZVxuICAgKiBsYXllcikuXG4gICAqL1xuICBvdmVycmlkZSBjb21wdXRlTWFzayhpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwgbWFzaz86IFRlbnNvcnxUZW5zb3JbXSk6IFRlbnNvclxuICAgICAgfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpbnB1dHMgPSBnZW5lcmljX3V0aWxzLnRvTGlzdChpbnB1dHMpO1xuICAgICAgbGV0IG1hc2tzOiBUZW5zb3JbXTtcbiAgICAgIGlmIChtYXNrID09IG51bGwpIHtcbiAgICAgICAgbWFza3MgPSBnZW5lcmljX3V0aWxzLnB5TGlzdFJlcGVhdChudWxsLCBpbnB1dHMubGVuZ3RoKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIG1hc2tzID0gZ2VuZXJpY191dGlscy50b0xpc3QobWFzayk7XG4gICAgICB9XG4gICAgICAvLyBUT0RPKG1pY2hhZWx0ZXJyeSk6IEFkZCBzdXBwb3J0IGZvciBtYXNrIGNhY2hpbmcuXG4gICAgICByZXR1cm4gdGhpcy5ydW5JbnRlcm5hbEdyYXBoKGlucHV0cywgbWFza3MpWzFdO1xuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBvdXRwdXQgc2hhcGUgb2YgdGhlIGxheWVyLlxuICAgKlxuICAgKiBBc3N1bWVzIHRoYXQgdGhlIGxheWVyIHdpbGwgYmUgYnVpbHQgdG8gbWF0Y2ggdGhhdCBpbnB1dCBzaGFwZSBwcm92aWRlZC5cbiAgICpcbiAgICogQHBhcmFtIGlucHV0U2hhcGUgQSBzaGFwZSAodHVwbGUgb2YgaW50ZWdlcnMpIG9yIGEgbGlzdCBvZiBzaGFwZSB0dXBsZXNcbiAgICogICAob25lIHBlciBvdXRwdXQgdGVuc29yIG9mIHRoZSBsYXllcikuIFNoYXBlIHR1cGxlcyBjYW4gaW5jbHVkZSBudWxsIGZvclxuICAgKiAgIGZyZWUgZGltZW5zaW9ucywgaW5zdGVhZCBvZiBhbiBpbnRlZ2VyLlxuICAgKi9cbiAgb3ZlcnJpZGUgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiBTaGFwZXxTaGFwZVtdIHtcbiAgICBjb25zdCBpbnB1dFNoYXBlcyA9IHR5cGVzX3V0aWxzLm5vcm1hbGl6ZVNoYXBlTGlzdChpbnB1dFNoYXBlKTtcbiAgICBpZiAoaW5wdXRTaGFwZXMubGVuZ3RoICE9PSB0aGlzLmlucHV0TGF5ZXJzLmxlbmd0aCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYEludmFsaWQgaW5wdXRTaGFwZSBhcmd1bWVudCAke2lucHV0U2hhcGV9OiBgICtcbiAgICAgICAgICBgbW9kZWwgaGFzICR7dGhpcy5pbnB1dExheWVycy5sZW5ndGh9IHRlbnNvciBpbnB1dHMuYCk7XG4gICAgfVxuXG4gICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBBZGQgY2FjaGluZ1xuICAgIGNvbnN0IGxheWVyc1RvT3V0cHV0U2hhcGVzOiB7W3NoYXBlS2V5OiBzdHJpbmddOiBTaGFwZX0gPSB7fTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGlucHV0U2hhcGVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBsYXllciA9IHRoaXMuaW5wdXRMYXllcnNbaV07XG4gICAgICBjb25zdCBpbnB1dFNoYXBlID0gaW5wdXRTaGFwZXNbaV07XG4gICAgICAvLyBJdCdzIGFuIGlucHV0IGxheWVyOiBjb21wdXRlT3V0cHV0U2hhcGUgaXMgaWRlbnRpdHksXG4gICAgICAvLyBhbmQgdGhlcmUgaXMgb25seSBvbmUgbm9kZSBhbmQgb25lIHRlbnNvciBvdXRwdXQuXG4gICAgICBjb25zdCBzaGFwZUtleSA9IGxheWVyLm5hbWUgKyAnXzBfMCc7XG4gICAgICBsYXllcnNUb091dHB1dFNoYXBlc1tzaGFwZUtleV0gPSBpbnB1dFNoYXBlO1xuICAgIH1cblxuICAgIGNvbnN0IGRlcHRoS2V5cyA9IE9iamVjdC5rZXlzKHRoaXMubm9kZXNCeURlcHRoKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAubWFwKHggPT4gcGFyc2VJbnQoeCwgMTApKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAuc29ydChnZW5lcmljX3V0aWxzLnJldmVyc2VOdW1iZXJDb21wYXJlKTtcbiAgICAvLyBJdGVyYXRlIG92ZXIgbm9kZXMsIGJ5IGRlcHRoIGxldmVsLlxuICAgIGlmIChkZXB0aEtleXMubGVuZ3RoID4gMSkge1xuICAgICAgZm9yIChjb25zdCBkZXB0aCBvZiBkZXB0aEtleXMpIHtcbiAgICAgICAgY29uc3Qgbm9kZXMgPSB0aGlzLm5vZGVzQnlEZXB0aFtkZXB0aF07XG4gICAgICAgIGZvciAoY29uc3Qgbm9kZSBvZiBub2Rlcykge1xuICAgICAgICAgIC8vIFRoaXMgaXMgYWx3YXlzIGEgc2luZ2xlIGxheWVyLCBuZXZlciBhIGxpc3QuXG4gICAgICAgICAgY29uc3QgbGF5ZXIgPSBub2RlLm91dGJvdW5kTGF5ZXI7XG4gICAgICAgICAgaWYgKHRoaXMuaW5wdXRMYXllcnMubWFwKHggPT4geC5pZCkuaW5kZXhPZihsYXllci5pZCkgIT09IC0xKSB7XG4gICAgICAgICAgICAvLyBXZSd2ZSBhbHJlYWR5IGNvdmVyZWQgdGhlIGlucHV0IGxheWVycyBhIGZldyBsaW5lcyBhYm92ZS5cbiAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgIH1cbiAgICAgICAgICAvLyBQb3RlbnRpYWxseSByZWR1bmRhbnQgbGlzdCwgc2FtZSBzaXplIG9mIG5vZGUuaW5wdXRUZW5zb3JzLlxuICAgICAgICAgIGNvbnN0IGlucHV0U2hhcGVzOiBTaGFwZVtdID0gW107XG4gICAgICAgICAgZm9yIChsZXQgaiA9IDA7IGogPCBub2RlLmluYm91bmRMYXllcnMubGVuZ3RoOyBqKyspIHtcbiAgICAgICAgICAgIGNvbnN0IGluYm91bmRMYXllciA9IG5vZGUuaW5ib3VuZExheWVyc1tqXTtcbiAgICAgICAgICAgIGNvbnN0IG5vZGVJbmRleCA9IG5vZGUubm9kZUluZGljZXNbal07XG4gICAgICAgICAgICBjb25zdCB0ZW5zb3JJbmRleCA9IG5vZGUudGVuc29ySW5kaWNlc1tqXTtcbiAgICAgICAgICAgIGNvbnN0IHNoYXBlS2V5ID0gYCR7aW5ib3VuZExheWVyLm5hbWV9XyR7bm9kZUluZGV4fV8ke3RlbnNvckluZGV4fWA7XG4gICAgICAgICAgICBjb25zdCBpbnB1dFNoYXBlID0gbGF5ZXJzVG9PdXRwdXRTaGFwZXNbc2hhcGVLZXldO1xuICAgICAgICAgICAgaW5wdXRTaGFwZXMucHVzaChpbnB1dFNoYXBlKTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGxheWVyLmNvbXB1dGVPdXRwdXRTaGFwZShcbiAgICAgICAgICAgICAgZ2VuZXJpY191dGlscy5zaW5nbGV0b25PckFycmF5KGlucHV0U2hhcGVzKSk7XG5cbiAgICAgICAgICBjb25zdCBvdXRwdXRTaGFwZXMgPSB0eXBlc191dGlscy5ub3JtYWxpemVTaGFwZUxpc3Qob3V0cHV0U2hhcGUpO1xuICAgICAgICAgIGNvbnN0IG5vZGVJbmRleCA9IGxheWVyLmluYm91bmROb2Rlcy5pbmRleE9mKG5vZGUpO1xuICAgICAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgb3V0cHV0U2hhcGVzLmxlbmd0aDsgaisrKSB7XG4gICAgICAgICAgICBjb25zdCBzaGFwZUtleSA9IGAke2xheWVyLm5hbWV9XyR7bm9kZUluZGV4fV8ke2p9YDtcbiAgICAgICAgICAgIGxheWVyc1RvT3V0cHV0U2hhcGVzW3NoYXBlS2V5XSA9IG91dHB1dFNoYXBlc1tqXTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBSZWFkIGZpbmFsIG91dHB1dCBzaGFwZXMgZnJvbSBsYXllcnNUb091dHB1dFNoYXBlcy5cbiAgICBjb25zdCBvdXRwdXRTaGFwZXM6IFNoYXBlW10gPSBbXTtcbiAgICBjb25zdCBvdXRwdXRTaGFwZUtleXM6IHN0cmluZ1tdID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLm91dHB1dExheWVycy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgbGF5ZXIgPSB0aGlzLm91dHB1dExheWVyc1tpXTtcbiAgICAgIGNvbnN0IG5vZGVJbmRleCA9IHRoaXMub3V0cHV0TGF5ZXJzTm9kZUluZGljZXNbaV07XG4gICAgICBjb25zdCB0ZW5zb3JJbmRleCA9IHRoaXMub3V0cHV0TGF5ZXJzVGVuc29ySW5kaWNlc1tpXTtcbiAgICAgIGNvbnN0IHNoYXBlS2V5ID0gYCR7bGF5ZXIubmFtZX1fJHtub2RlSW5kZXh9XyR7dGVuc29ySW5kZXh9YDtcbiAgICAgIG91dHB1dFNoYXBlS2V5cy5wdXNoKHNoYXBlS2V5KTtcbiAgICB9XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG91dHB1dFNoYXBlS2V5cy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3Qga2V5ID0gb3V0cHV0U2hhcGVLZXlzW2ldO1xuICAgICAgZ2VuZXJpY191dGlscy5hc3NlcnQoa2V5IGluIGxheWVyc1RvT3V0cHV0U2hhcGVzKTtcbiAgICAgIG91dHB1dFNoYXBlcy5wdXNoKGxheWVyc1RvT3V0cHV0U2hhcGVzW2tleV0pO1xuICAgIH1cblxuICAgIC8vIFRPRE8obWljaGFlbHRlcnJ5KTogVXBkYXRlIGNhY2hlXG4gICAgcmV0dXJuIGdlbmVyaWNfdXRpbHMuc2luZ2xldG9uT3JBcnJheShvdXRwdXRTaGFwZXMpO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIG91dHB1dCB0ZW5zb3JzIGZvciBuZXcgaW5wdXRzLlxuICAgKlxuICAgKiBOb3RlOlxuICAgKiAgIC0gRXhwZWN0cyBgaW5wdXRzYCB0byBiZSBhIGxpc3QgKHBvdGVudGlhbGx5IHdpdGggMSBlbGVtZW50KS5cbiAgICpcbiAgICogQHBhcmFtIGlucHV0cyBMaXN0IG9mIHRlbnNvcnNcbiAgICogQHBhcmFtIG1hc2tzIExpc3Qgb2YgbWFza3MgKHRlbnNvcnMgb3IgbnVsbCkuXG4gICAqIEByZXR1cm4gVGhyZWUgbGlzdHM6IG91dHB1dFRlbnNvcnMsIG91dHB1dE1hc2tzLCBvdXRwdXRTaGFwZXNcbiAgICovXG4gIHByb3RlY3RlZCBydW5JbnRlcm5hbEdyYXBoKGlucHV0czogVGVuc29yW10sIG1hc2tzPzogVGVuc29yW10pOlxuICAgICAgW1RlbnNvcltdLCBUZW5zb3JbXSwgU2hhcGVbXV0ge1xuICAgIGlmIChtYXNrcyA9PSBudWxsKSB7XG4gICAgICBtYXNrcyA9IGdlbmVyaWNfdXRpbHMucHlMaXN0UmVwZWF0KG51bGwsIGlucHV0cy5sZW5ndGgpO1xuICAgIH1cblxuICAgIC8vIERpY3Rpb25hcnkgbWFwcGluZyByZWZlcmVuY2UgdGVuc29ycyB0byB0dXBsZXNcbiAgICAvLyAoY29tcHV0ZWQgdGVuc29yLCBjb21wdXRlIG1hc2spXG4gICAgLy8gd2UgYXNzdW1lIGEgMToxIG1hcHBpbmcgZnJvbSB0ZW5zb3IgdG8gbWFza1xuICAgIC8vIFRPRE86IHJhaXNlIGV4Y2VwdGlvbiB3aGVuIGEgYC5jb21wdXRlTWFzaygpYCBjYWxsXG4gICAgLy8gZG9lcyBub3QgcmV0dXJuIGEgbGlzdCB0aGUgc2FtZSBzaXplIGFzIGBjYWxsYFxuICAgIGNvbnN0IHRlbnNvck1hcDoge1t0ZW5zb3JJRDogc3RyaW5nXTogW1RlbnNvciwgVGVuc29yXX0gPSB7fTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuaW5wdXRzLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCB4ID0gdGhpcy5pbnB1dHNbaV07XG4gICAgICBjb25zdCB5ID0gaW5wdXRzW2ldO1xuICAgICAgY29uc3QgbWFzayA9IG1hc2tzW2ldO1xuICAgICAgdGVuc29yTWFwW3guaWRdID0gW3ksIG1hc2tdO1xuICAgIH1cblxuICAgIGNvbnN0IGRlcHRoS2V5cyA9IE9iamVjdC5rZXlzKHRoaXMubm9kZXNCeURlcHRoKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAubWFwKHggPT4gcGFyc2VJbnQoeCwgMTApKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAuc29ydChnZW5lcmljX3V0aWxzLnJldmVyc2VOdW1iZXJDb21wYXJlKTtcbiAgICBmb3IgKGNvbnN0IGRlcHRoIG9mIGRlcHRoS2V5cykge1xuICAgICAgY29uc3Qgbm9kZXMgPSB0aGlzLm5vZGVzQnlEZXB0aFtkZXB0aF07XG4gICAgICBmb3IgKGNvbnN0IG5vZGUgb2Ygbm9kZXMpIHtcbiAgICAgICAgLy8gVGhpcyBpcyBhbHdheXMgYSBzaW5nbGUgbGF5ZXIsIG5ldmVyIGEgbGlzdC5cbiAgICAgICAgY29uc3QgbGF5ZXIgPSBub2RlLm91dGJvdW5kTGF5ZXI7XG4gICAgICAgIGNvbnN0IHJlZmVyZW5jZUlucHV0VGVuc29ycyA9IG5vZGUuaW5wdXRUZW5zb3JzO1xuICAgICAgICBjb25zdCByZWZlcmVuY2VPdXRwdXRUZW5zb3JzID0gbm9kZS5vdXRwdXRUZW5zb3JzO1xuXG4gICAgICAgIC8vIElmIGFsbCBwcmV2aW91cyBpbnB1dCB0ZW5zb3JzIGFyZSBhdmFpbGFibGUgaW4gdGVuc29yTWFwLFxuICAgICAgICAvLyB0aGVuIGNhbGwgbm9kZS5pbmJvdW5kTGF5ZXIgb24gdGhlbS5cbiAgICAgICAgLy8gTGlzdCBvZiB0dXBsZXMgW2lucHV0LCBtYXNrXTpcbiAgICAgICAgY29uc3QgY29tcHV0ZWREYXRhID0gbmV3IEFycmF5PFtUZW5zb3IsIFRlbnNvcl0+KCk7XG4gICAgICAgIGZvciAoY29uc3QgeCBvZiByZWZlcmVuY2VJbnB1dFRlbnNvcnMpIHtcbiAgICAgICAgICBpZiAoeC5pZCBpbiB0ZW5zb3JNYXApIHtcbiAgICAgICAgICAgIGNvbXB1dGVkRGF0YS5wdXNoKHRlbnNvck1hcFt4LmlkXSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGlmIChjb21wdXRlZERhdGEubGVuZ3RoID09PSByZWZlcmVuY2VJbnB1dFRlbnNvcnMubGVuZ3RoKSB7XG4gICAgICAgICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBBZGQgSy5uYW1lX3Njb3BlIGhlcmUsIGlmIHdlIG5lZWQgaXQuXG4gICAgICAgICAgbGV0IGt3YXJnczogS3dhcmdzID0ge307XG4gICAgICAgICAgbGV0IGNvbXB1dGVkVGVuc29yczogVGVuc29yW107XG4gICAgICAgICAgbGV0IGNvbXB1dGVkTWFza3M6IFRlbnNvcltdO1xuICAgICAgICAgIGxldCBvdXRwdXRUZW5zb3JzOiBUZW5zb3JbXTtcbiAgICAgICAgICBsZXQgb3V0cHV0TWFza3M6IFRlbnNvcltdO1xuICAgICAgICAgIC8vIGNhbGwgbGF5ZXJcbiAgICAgICAgICBpZiAobm9kZS5jYWxsQXJncyAhPSBudWxsKSB7XG4gICAgICAgICAgICBrd2FyZ3MgPSBub2RlLmNhbGxBcmdzO1xuICAgICAgICAgIH1cbiAgICAgICAgICBpZiAoY29tcHV0ZWREYXRhLmxlbmd0aCA9PT0gMSkge1xuICAgICAgICAgICAgY29uc3QgW2NvbXB1dGVkVGVuc29yLCBjb21wdXRlZE1hc2tdID0gY29tcHV0ZWREYXRhWzBdO1xuICAgICAgICAgICAgaWYgKGt3YXJnc1snbWFzayddID09IG51bGwpIHtcbiAgICAgICAgICAgICAga3dhcmdzWydtYXNrJ10gPSBjb21wdXRlZE1hc2s7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBvdXRwdXRUZW5zb3JzID1cbiAgICAgICAgICAgICAgICBnZW5lcmljX3V0aWxzLnRvTGlzdChsYXllci5jYWxsKGNvbXB1dGVkVGVuc29yLCBrd2FyZ3MpKTtcbiAgICAgICAgICAgIG91dHB1dE1hc2tzID0gZ2VuZXJpY191dGlscy50b0xpc3QoXG4gICAgICAgICAgICAgICAgbGF5ZXIuY29tcHV0ZU1hc2soY29tcHV0ZWRUZW5zb3IsIGNvbXB1dGVkTWFzaykpO1xuICAgICAgICAgICAgY29tcHV0ZWRUZW5zb3JzID0gW2NvbXB1dGVkVGVuc29yXTtcbiAgICAgICAgICAgIGNvbXB1dGVkTWFza3MgPSBbY29tcHV0ZWRNYXNrXTtcbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgY29tcHV0ZWRUZW5zb3JzID0gY29tcHV0ZWREYXRhLm1hcCh4ID0+IHhbMF0pO1xuICAgICAgICAgICAgY29tcHV0ZWRNYXNrcyA9IGNvbXB1dGVkRGF0YS5tYXAoeCA9PiB4WzFdKTtcbiAgICAgICAgICAgIGlmIChrd2FyZ3NbJ21hc2snXSA9PSBudWxsKSB7XG4gICAgICAgICAgICAgIGt3YXJnc1snbWFzayddID0gY29tcHV0ZWRNYXNrcztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIG91dHB1dFRlbnNvcnMgPVxuICAgICAgICAgICAgICAgIGdlbmVyaWNfdXRpbHMudG9MaXN0KGxheWVyLmNhbGwoY29tcHV0ZWRUZW5zb3JzLCBrd2FyZ3MpKTtcbiAgICAgICAgICAgIG91dHB1dE1hc2tzID0gZ2VuZXJpY191dGlscy50b0xpc3QoXG4gICAgICAgICAgICAgICAgbGF5ZXIuY29tcHV0ZU1hc2soY29tcHV0ZWRUZW5zb3JzLCBjb21wdXRlZE1hc2tzKSk7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgaWYgKGxheWVyLmFjdGl2aXR5UmVndWxhcml6ZXIpIHtcbiAgICAgICAgICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKFxuICAgICAgICAgICAgICAgICdMYXllcnNNb2RlbCBpbnZvY2F0aW9uIHdpdGggY29uY3JldGUgVGVuc29yIHZhbHVlKHMpIGluIHRoZSAnICtcbiAgICAgICAgICAgICAgICAncHJlc2VuY2Ugb2YgYWN0aXZpdHkgcmVndWxhcml6ZXIocykgaXMgbm90IHN1cHBvcnRlZCB5ZXQuJyk7XG4gICAgICAgICAgfVxuICAgICAgICAgIC8vIFRPRE8obWljaGFlbHRlcnJ5KTogQWRkIG1vZGVsIHVwZGF0ZXMgYW5kIGxvc3Nlc1xuXG4gICAgICAgICAgLy8gVXBkYXRlIHRlbnNvciBtYXAuXG4gICAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCByZWZlcmVuY2VPdXRwdXRUZW5zb3JzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgICAgICBjb25zdCB4ID0gcmVmZXJlbmNlT3V0cHV0VGVuc29yc1tpXTtcbiAgICAgICAgICAgIGNvbnN0IHkgPSBvdXRwdXRUZW5zb3JzW2ldO1xuICAgICAgICAgICAgY29uc3QgbWFzayA9IG91dHB1dE1hc2tzW2ldO1xuICAgICAgICAgICAgdGVuc29yTWFwW3guaWRdID0gW3ksIG1hc2tdO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIGNvbnN0IG91dHB1dFRlbnNvcnM6IFRlbnNvcltdID0gW107XG4gICAgY29uc3Qgb3V0cHV0TWFza3M6IFRlbnNvcltdID0gW107XG4gICAgY29uc3Qgb3V0cHV0U2hhcGVzOiBTaGFwZVtdID0gW107XG4gICAgZm9yIChjb25zdCB4IG9mIHRoaXMub3V0cHV0cykge1xuICAgICAgZ2VuZXJpY191dGlscy5hc3NlcnQoXG4gICAgICAgICAgeC5pZCBpbiB0ZW5zb3JNYXAsIGBDb3VsZCBub3QgY29tcHV0ZSBvdXRwdXQgJHt4Lm5hbWV9IDogJHt4LmlkfWApO1xuICAgICAgY29uc3QgW3RlbnNvciwgbWFza10gPSB0ZW5zb3JNYXBbeC5pZF07XG4gICAgICBvdXRwdXRTaGFwZXMucHVzaCh0ZW5zb3Iuc2hhcGUpO1xuICAgICAgb3V0cHV0VGVuc29ycy5wdXNoKHRlbnNvcik7XG4gICAgICBvdXRwdXRNYXNrcy5wdXNoKG1hc2spO1xuICAgIH1cblxuICAgIC8vIFRPRE8obWljaGFlbHRlcnJ5KTogQWRkIHN1cHBvcnQgZm9yIGNhY2hlcy5cbiAgICByZXR1cm4gW291dHB1dFRlbnNvcnMsIG91dHB1dE1hc2tzLCBvdXRwdXRTaGFwZXNdO1xuICB9XG5cbiAgLyoqXG4gICAqIEJ1aWxkcyBhIG1hcCBvZiBpbnRlcm5hbCBub2RlIGtleXMgdG8gbm9kZSBvcmRlcmluZy5cbiAgICogVXNlZCBpbiBzZXJpYWxpemFpb24gYSBub2RlIG9yZGVyaW5ncyBtYXkgY2hhbmdlIGFzIHVudXNlZCBub2RlcyBhcmVcbiAgICogZHJvcHBlZC4gUG9ydGluZyBOb3RlOiAgVGhpcyBoZWxwZXIgbWV0aG9kIHdhcyBwdWxsZWQgb3V0IG9mIGdldENvbmZpZyB0b1xuICAgKiBpbXByb3ZlIHJlYWRhYmlsaXR5LlxuICAgKiBAcGFyYW0gbGF5ZXJzIEFuIGFycmF5IG9mIExheWVycyBpbiB0aGUgbW9kZWwuXG4gICAqIEByZXR1cm5zIE1hcCBvZiBOb2RlIEtleXMgdG8gaW5kZXggb3JkZXIgd2l0aGluIHRoZSBsYXllci5cbiAgICovXG4gIHByaXZhdGUgYnVpbGROb2RlQ29udmVyc2lvbk1hcChsYXllcnM6IExheWVyW10pOiB7W25vZGVLZXk6IHN0cmluZ106IG51bWJlcn0ge1xuICAgIGNvbnN0IG5vZGVDb252ZXJzaW9uTWFwOiB7W25vZGVLZXk6IHN0cmluZ106IG51bWJlcn0gPSB7fTtcbiAgICBsZXQga2VwdE5vZGVzOiBudW1iZXI7XG4gICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLmxheWVycykge1xuICAgICAga2VwdE5vZGVzID0gbGF5ZXIgaW5zdGFuY2VvZiBDb250YWluZXIgPyAxIDogMDtcbiAgICAgIGZvciAobGV0IG9yaWdpbmFsTm9kZUluZGV4ID0gMDtcbiAgICAgICAgICAgb3JpZ2luYWxOb2RlSW5kZXggPCBsYXllci5pbmJvdW5kTm9kZXMubGVuZ3RoOyBvcmlnaW5hbE5vZGVJbmRleCsrKSB7XG4gICAgICAgIGNvbnN0IG5vZGVLZXkgPSBDb250YWluZXIubm9kZUtleShsYXllciwgb3JpZ2luYWxOb2RlSW5kZXgpO1xuICAgICAgICBpZiAodGhpcy5jb250YWluZXJOb2Rlcy5oYXMobm9kZUtleSkpIHtcbiAgICAgICAgICAvLyBpLmUuIHdlIG1hcmsgaXQgdG8gYmUgc2F2ZWRcbiAgICAgICAgICBub2RlQ29udmVyc2lvbk1hcFtub2RlS2V5XSA9IGtlcHROb2RlcztcbiAgICAgICAgICBrZXB0Tm9kZXMgKz0gMTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gbm9kZUNvbnZlcnNpb25NYXA7XG4gIH1cblxuICAvKipcbiAgICogUmV0cmlldmVzIGEgbGF5ZXIgYmFzZWQgb24gZWl0aGVyIGl0cyBuYW1lICh1bmlxdWUpIG9yIGluZGV4LlxuICAgKlxuICAgKiBJbmRpY2VzIGFyZSBiYXNlZCBvbiBvcmRlciBvZiBob3Jpem9udGFsIGdyYXBoIHRyYXZlcnNhbCAoYm90dG9tLXVwKS5cbiAgICpcbiAgICogSWYgYm90aCBgbmFtZWAgYW5kIGBpbmRleGAgYXJlIHNwZWNpZmllZCwgYGluZGV4YCB0YWtlcyBwcmVjZWRlbmNlLlxuICAgKlxuICAgKiBAcGFyYW0gbmFtZSBOYW1lIG9mIGxheWVyLlxuICAgKiBAcGFyYW0gaW5kZXggSW5kZXggb2YgbGF5ZXIuXG4gICAqIEByZXR1cm5zIEEgTGF5ZXIgaW5zdGFuY2UuXG4gICAqIEB0aHJvd3MgVmFsdWVFcnJvcjogSW4gY2FzZSBvZiBpbnZhbGlkIGxheWVyIG5hbWUgb3IgaW5kZXguXG4gICAqXG4gICAqIEBkb2Mge1xuICAgKiAgICBoZWFkaW5nOiAnTGF5ZXJzJyxcbiAgICogICAgc3ViaGVhZGluZzogJ0NsYXNzZXMnLFxuICAgKiAgICBuYW1lc3BhY2U6ICdsYXllcnMnLFxuICAgKiAgICBzdWJjbGFzc2VzOiBbJ0xheWVyc01vZGVsJ11cbiAgICogfVxuICAgKi9cbiAgZ2V0TGF5ZXIobmFtZT86IHN0cmluZywgaW5kZXg/OiBudW1iZXIpOiBMYXllciB7XG4gICAgaWYgKGluZGV4ICE9IG51bGwpIHtcbiAgICAgIGlmICh0aGlzLmxheWVycy5sZW5ndGggPD0gaW5kZXgpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgV2FzIGFza2VkIHRvIHJldHJpZXZlIGxheWVyIGF0IGluZGV4ICR7aW5kZXh9LCBidXQgbW9kZWwgb25seSBgICtcbiAgICAgICAgICAgIGBoYXMgJHt0aGlzLmxheWVycy5sZW5ndGh9IGxheWVyKHMpLmApO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIHRoaXMubGF5ZXJzW2luZGV4XTtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgaWYgKG5hbWUgPT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcignUHJvdmlkZSBlaXRoZXIgYSBsYXllciBuYW1lIG9yIGxheWVyIGluZGV4Jyk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLmxheWVycykge1xuICAgICAgaWYgKGxheWVyLm5hbWUgPT09IG5hbWUpIHtcbiAgICAgICAgcmV0dXJuIGxheWVyO1xuICAgICAgfVxuICAgIH1cbiAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihgTm8gc3VjaCBsYXllcjogJHtuYW1lfWApO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHJpZXZlcyB0aGUgQ29udGFpbmVyJ3MgY3VycmVudCBsb3NzIHZhbHVlcy5cbiAgICpcbiAgICogVXNlZCBmb3IgcmVndWxhcml6ZXJzIGR1cmluZyB0cmFpbmluZy5cbiAgICovXG4gIG92ZXJyaWRlIGNhbGN1bGF0ZUxvc3NlcygpOiBTY2FsYXJbXSB7XG4gICAgLy8gUG9ydGluZyBOb2RlOiBUaGlzIGlzIGFuIGF1Z21lbnRhdGlvbiB0byBDb250YWluZXIubG9zcyBpbiBQeUtlcmFzLlxuICAgIC8vICAgSW4gUHlLZXJhcywgQ29udGFpbmVyLmxvc3MgcmV0dXJucyBzeW1ib2xpYyB0ZW5zb3JzLiBIZXJlIGEgY29uY3JldGVcbiAgICAvLyAgIFRlbnNvciAoc3BlY2lmaWNhbGx5IFNjYWxhcikgdmFsdWVzIGFyZSByZXR1cm5lZC4gVGhpcyBpcyBkdWUgdG8gdGhlXG4gICAgLy8gICBpbXBlcmF0aXZlIGJhY2tlbmQuXG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgbG9zc2VzOiBTY2FsYXJbXSA9IFtdO1xuICAgICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLmxheWVycykge1xuICAgICAgICBmb3IgKGxldCBub2RlSW5kZXggPSAwOyBub2RlSW5kZXggPCBsYXllci5pbmJvdW5kTm9kZXMubGVuZ3RoO1xuICAgICAgICAgICAgICsrbm9kZUluZGV4KSB7XG4gICAgICAgICAgY29uc3Qgbm9kZUtleSA9IENvbnRhaW5lci5ub2RlS2V5KGxheWVyLCBub2RlSW5kZXgpO1xuICAgICAgICAgIGlmICh0aGlzLmNvbnRhaW5lck5vZGVzLmhhcyhub2RlS2V5KSkge1xuICAgICAgICAgICAgbG9zc2VzLnB1c2goLi4ubGF5ZXIuY2FsY3VsYXRlTG9zc2VzKCkpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgLy8gVE9ETyhjYWlzKTogQWRkIGFueSB1bmNvbmRpdGlvbmFsIG1vZGVsLWxldmVsIGxvc3Nlcz9cbiAgICAgIHJldHVybiBsb3NzZXM7XG4gICAgfSk7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtuYW1lOiB0aGlzLm5hbWV9O1xuXG4gICAgLy8gQnVpbGQgYSBtYXAgZnJvbSBsYXllciB1bmlxdWUgbmFtZSAoc2VsZi5fbm9kZV9rZXkpXG4gICAgLy8gdG8gdGhlIGluZGV4IG9mIHRoZSBub2RlcyB0aGF0IGFyZSBzYXZlZCBpbiB0aGUgY29uZmlnLlxuICAgIC8vIE9ubHkgbm9kZXMgaW4gY29udGFpbmVyX25vZGVzIGFyZSBzYXZlZC5cbiAgICBjb25zdCBub2RlQ29udmVyc2lvbk1hcDoge1tub2RlS2V5OiBzdHJpbmddOiBudW1iZXJ9ID1cbiAgICAgICAgdGhpcy5idWlsZE5vZGVDb252ZXJzaW9uTWFwKHRoaXMubGF5ZXJzKTtcblxuICAgIC8vIFNlcmlhbGl6ZSBhbmQgc2F2ZSB0aGUgbGF5ZXJzIGluIGxheWVyQ29uZmlnc1xuICAgIGNvbnN0IGxheWVyQ29uZmlncyA9IFtdO1xuICAgIGZvciAoY29uc3QgbGF5ZXIgb2YgdGhpcy5sYXllcnMpIHtcbiAgICAgIGNvbnN0IGxheWVyQ2xhc3NOYW1lID0gbGF5ZXIuZ2V0Q2xhc3NOYW1lKCk7XG4gICAgICBjb25zdCBsYXllckNvbmZpZyA9IGxheWVyLmdldENvbmZpZygpO1xuICAgICAgY29uc3QgZmlsdGVyZWRJbmJvdW5kTm9kZXMgPSBbXTtcbiAgICAgIGZvciAobGV0IG9yaWdpbmFsTm9kZUluZGV4ID0gMDtcbiAgICAgICAgICAgb3JpZ2luYWxOb2RlSW5kZXggPCBsYXllci5pbmJvdW5kTm9kZXMubGVuZ3RoOyBvcmlnaW5hbE5vZGVJbmRleCsrKSB7XG4gICAgICAgIGNvbnN0IG5vZGUgPSBsYXllci5pbmJvdW5kTm9kZXNbb3JpZ2luYWxOb2RlSW5kZXhdO1xuICAgICAgICBjb25zdCBub2RlS2V5ID0gQ29udGFpbmVyLm5vZGVLZXkobGF5ZXIsIG9yaWdpbmFsTm9kZUluZGV4KTtcbiAgICAgICAgbGV0IGt3YXJncyA9IHt9O1xuICAgICAgICBpZiAodGhpcy5jb250YWluZXJOb2Rlcy5oYXMobm9kZUtleSkpIHtcbiAgICAgICAgICAvLyBUaGUgbm9kZSBpcyByZWxldmFudCB0byB0aGUgbW9kZWw6XG4gICAgICAgICAgLy8gYWRkIHRvIGZpbHRlcmVkSW5ib3VuZE5vZGVzLlxuICAgICAgICAgIGlmIChub2RlLmNhbGxBcmdzKSB7XG4gICAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgICBKU09OLnN0cmluZ2lmeShub2RlLmNhbGxBcmdzKTtcbiAgICAgICAgICAgICAga3dhcmdzID0gbm9kZS5jYWxsQXJncztcbiAgICAgICAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICAgICAgICBgTGF5ZXIgJHtsYXllci5uYW1lfSB3YXMgcGFzc2VkIGAgK1xuICAgICAgICAgICAgICAgICAgYG5vbi1zZXJpYWxpemFibGUga2V5d29yZCBhcmd1bWVudHM6IGAgK1xuICAgICAgICAgICAgICAgICAgYCR7bm9kZS5jYWxsQXJnc30uIFRoZXkgd2lsbCBub3QgYmUgaW5jbHVkZWQgYCArXG4gICAgICAgICAgICAgICAgICBgaW4gdGhlIHNlcmlhbGl6ZWQgbW9kZWwgKGFuZCB0aHVzIHdpbGwgYmUgYCArXG4gICAgICAgICAgICAgICAgICBgbWlzc2luZyBhdCBkZXNlcmlhbGl6YXRpb24gdGltZSkuYCk7XG4gICAgICAgICAgICAgIGt3YXJncyA9IHt9O1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgICBpZiAobm9kZS5pbmJvdW5kTGF5ZXJzLmxlbmd0aCA+IDApIHtcbiAgICAgICAgICAgIGNvbnN0IG5vZGVEYXRhID0gW107XG4gICAgICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5vZGUuaW5ib3VuZExheWVycy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgICBjb25zdCBpbmJvdW5kTGF5ZXIgPSBub2RlLmluYm91bmRMYXllcnNbaV07XG4gICAgICAgICAgICAgIGNvbnN0IG5vZGVJbmRleCA9IG5vZGUubm9kZUluZGljZXNbaV07XG4gICAgICAgICAgICAgIGNvbnN0IHRlbnNvckluZGV4ID0gbm9kZS50ZW5zb3JJbmRpY2VzW2ldO1xuICAgICAgICAgICAgICBjb25zdCBub2RlS2V5ID0gQ29udGFpbmVyLm5vZGVLZXkoaW5ib3VuZExheWVyLCBub2RlSW5kZXgpO1xuICAgICAgICAgICAgICBsZXQgbmV3Tm9kZUluZGV4ID0gbm9kZUNvbnZlcnNpb25NYXBbbm9kZUtleV07XG4gICAgICAgICAgICAgIGlmIChuZXdOb2RlSW5kZXggPT0gbnVsbCkge1xuICAgICAgICAgICAgICAgIG5ld05vZGVJbmRleCA9IDA7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgbm9kZURhdGEucHVzaChcbiAgICAgICAgICAgICAgICAgIFtpbmJvdW5kTGF5ZXIubmFtZSwgbmV3Tm9kZUluZGV4LCB0ZW5zb3JJbmRleCwga3dhcmdzXSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBmaWx0ZXJlZEluYm91bmROb2Rlcy5wdXNoKG5vZGVEYXRhKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGNvbnN0IGRpY3Q6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHt9O1xuICAgICAgZGljdFsnbmFtZSddID0gbGF5ZXIubmFtZTtcbiAgICAgIGRpY3RbJ2NsYXNzTmFtZSddID0gbGF5ZXJDbGFzc05hbWU7XG4gICAgICBkaWN0Wydjb25maWcnXSA9IGxheWVyQ29uZmlnO1xuICAgICAgZGljdFsnaW5ib3VuZE5vZGVzJ10gPSBmaWx0ZXJlZEluYm91bmROb2RlcztcbiAgICAgIGxheWVyQ29uZmlncy5wdXNoKGRpY3QpO1xuICAgIH1cbiAgICBjb25maWdbJ2xheWVycyddID0gbGF5ZXJDb25maWdzO1xuICAgIC8vIEdhdGhlciBpbmZvIGFib3V0IGlucHV0cyBhbmQgb3V0cHV0c1xuICAgIGNvbnN0IG1vZGVsSW5wdXRzID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLmlucHV0TGF5ZXJzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBsYXllciA9IHRoaXMuaW5wdXRMYXllcnNbaV07XG4gICAgICBjb25zdCBub2RlSW5kZXggPSB0aGlzLmlucHV0TGF5ZXJzTm9kZUluZGljZXNbaV07XG5cbiAgICAgIGNvbnN0IG5vZGVLZXkgPSBDb250YWluZXIubm9kZUtleShsYXllciwgbm9kZUluZGV4KTtcbiAgICAgIGlmICghdGhpcy5jb250YWluZXJOb2Rlcy5oYXMobm9kZUtleSkpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBsZXQgbmV3Tm9kZUluZGV4ID0gbm9kZUNvbnZlcnNpb25NYXBbbm9kZUtleV07XG4gICAgICBpZiAobmV3Tm9kZUluZGV4ID09PSBudWxsIHx8IG5ld05vZGVJbmRleCA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICAgIG5ld05vZGVJbmRleCA9IDA7XG4gICAgICB9XG4gICAgICBjb25zdCB0ZW5zb3JJbmRleCA9IHRoaXMuaW5wdXRMYXllcnNUZW5zb3JJbmRpY2VzW2ldO1xuICAgICAgbW9kZWxJbnB1dHMucHVzaChbbGF5ZXIubmFtZSwgbmV3Tm9kZUluZGV4LCB0ZW5zb3JJbmRleF0pO1xuICAgIH1cbiAgICBjb25maWdbJ2lucHV0TGF5ZXJzJ10gPSBtb2RlbElucHV0cztcblxuICAgIGNvbnN0IG1vZGVsT3V0cHV0cyA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5vdXRwdXRMYXllcnMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IGxheWVyID0gdGhpcy5vdXRwdXRMYXllcnNbaV07XG4gICAgICBjb25zdCBub2RlSW5kZXggPSB0aGlzLm91dHB1dExheWVyc05vZGVJbmRpY2VzW2ldO1xuXG4gICAgICBjb25zdCBub2RlS2V5ID0gQ29udGFpbmVyLm5vZGVLZXkobGF5ZXIsIG5vZGVJbmRleCk7XG4gICAgICBpZiAoIXRoaXMuY29udGFpbmVyTm9kZXMuaGFzKG5vZGVLZXkpKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgbGV0IG5ld05vZGVJbmRleCA9IG5vZGVDb252ZXJzaW9uTWFwW25vZGVLZXldO1xuICAgICAgaWYgKG5ld05vZGVJbmRleCA9PT0gbnVsbCB8fCBuZXdOb2RlSW5kZXggPT09IHVuZGVmaW5lZCkge1xuICAgICAgICBuZXdOb2RlSW5kZXggPSAwO1xuICAgICAgfVxuICAgICAgY29uc3QgdGVuc29ySW5kZXggPSB0aGlzLm91dHB1dExheWVyc1RlbnNvckluZGljZXNbaV07XG4gICAgICBtb2RlbE91dHB1dHMucHVzaChbbGF5ZXIubmFtZSwgbmV3Tm9kZUluZGV4LCB0ZW5zb3JJbmRleF0pO1xuICAgIH1cbiAgICBjb25maWdbJ291dHB1dExheWVycyddID0gbW9kZWxPdXRwdXRzO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cblxuICAvKipcbiAgICogSW5zdGFudGlhdGVzIGEgTGF5ZXJzTW9kZWwgZnJvbSBpdHMgY29uZmlnIChvdXRwdXQgb2YgYGdldF9jb25maWcoKWApLlxuICAgKiBAcGFyYW0gY2xzIHRoZSBjbGFzcyB0byBjcmVhdGVcbiAgICogQHBhcmFtIGNvbmZpZyBMYXllcnNNb2RlbCBjb25maWcgZGljdGlvbmFyeS5cbiAgICogQHBhcmFtIGN1c3RvbU9iamVjdHMgQW4gb3B0aW9uYWwgZGljdGlvbmFyeSBvZiBjdXN0b20gb2JqZWN0cy5cbiAgICogQHBhcmFtIGZhc3RXZWlnaHRJbml0IE9wdGlvbmFsIGZsYWcgdG8gdXNlIGZhc3Qgd2VpZ2h0IGluaXRpYWxpemF0aW9uXG4gICAqICAgZHVyaW5nIGRlc2VyaWFsaXphdGlvbi4gVGhpcyBpcyBhcHBsaWNhYmxlIHRvIGNhc2VzIGluIHdoaWNoXG4gICAqICAgdGhlIGluaXRpYWxpemF0aW9uIHdpbGwgYmUgaW1tZWRpYXRlbHkgb3ZlcndyaXR0ZW4gYnkgbG9hZGVkIHdlaWdodFxuICAgKiAgIHZhbHVlcy4gRGVmYXVsdDogYGZhbHNlYC5cbiAgICogQHJldHVybnMgQSBMYXllcnNNb2RlbCBpbnN0YW5jZS5cbiAgICogQHRocm93cyBWYWx1ZUVycm9yOiBJbiBjYXNlIG9mIGltcHJvcGVybHkgZm9ybWF0dGVkIGNvbmZpZyBkaWN0LlxuICAgKi9cbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBmcm9tQ29uZmlnPFQgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZT4oXG4gICAgICBjbHM6IHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlQ29uc3RydWN0b3I8VD4sXG4gICAgICBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCxcbiAgICAgIGN1c3RvbU9iamVjdHMgPSB7fSBhcyBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QsXG4gICAgICBmYXN0V2VpZ2h0SW5pdCA9IGZhbHNlKTogVCB7XG4gICAgLy8gTGF5ZXIgaW5zdGFuY2VzIGNyZWF0ZWQgZHVyaW5nXG4gICAgLy8gdGhlIGdyYXBoIHJlY29uc3RydWN0aW9uIHByb2Nlc3NcbiAgICBjb25zdCBjcmVhdGVkTGF5ZXJzOiB7W2xheWVyTmFtZTogc3RyaW5nXTogTGF5ZXJ9ID0ge307XG5cbiAgICAvLyBEaWN0aW9uYXJ5IG1hcHBpbmcgbGF5ZXIgaW5zdGFuY2VzIHRvXG4gICAgLy8gbm9kZSBkYXRhIHRoYXQgc3BlY2lmaWVzIGEgbGF5ZXIgY2FsbC5cbiAgICAvLyBJdCBhY3RzIGFzIGEgcXVldWUgdGhhdCBtYWludGFpbnMgYW55IHVucHJvY2Vzc2VkXG4gICAgLy8gbGF5ZXIgY2FsbCB1bnRpbCBpdCBiZWNvbWVzIHBvc3NpYmxlIHRvIHByb2Nlc3MgaXRcbiAgICAvLyAoaS5lLiB1bnRpbCB0aGUgaW5wdXQgdGVuc29ycyB0byB0aGUgY2FsbCBhbGwgZXhpc3QpLlxuICAgIGNvbnN0IHVucHJvY2Vzc2VkTm9kZXM6IHtbbGF5ZXI6IHN0cmluZ106IFRlbnNvcktleVdpdGhBcmdzQXJyYXlbXVtdfSA9IHt9O1xuICAgIGZ1bmN0aW9uIGFkZFVucHJvY2Vzc2VkTm9kZShcbiAgICAgICAgbGF5ZXI6IExheWVyLCBub2RlRGF0YTogVGVuc29yS2V5V2l0aEFyZ3NBcnJheVtdKSB7XG4gICAgICBpZiAoIShsYXllci5uYW1lIGluIHVucHJvY2Vzc2VkTm9kZXMpKSB7XG4gICAgICAgIHVucHJvY2Vzc2VkTm9kZXNbbGF5ZXIubmFtZV0gPSBbbm9kZURhdGFdO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdW5wcm9jZXNzZWROb2Rlc1tsYXllci5uYW1lXS5wdXNoKG5vZGVEYXRhKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICBmdW5jdGlvbiBwcm9jZXNzTm9kZShsYXllcjogTGF5ZXIsIG5vZGVEYXRhOiBUZW5zb3JLZXlXaXRoQXJnc0FycmF5W10pIHtcbiAgICAgIGNvbnN0IGlucHV0VGVuc29yczogU3ltYm9saWNUZW5zb3JbXSA9IFtdO1xuICAgICAgbGV0IGt3YXJncztcbiAgICAgIGZvciAoY29uc3QgaW5wdXREYXRhIG9mIG5vZGVEYXRhKSB7XG4gICAgICAgIGNvbnN0IGluYm91bmRMYXllck5hbWUgPSBpbnB1dERhdGFbMF07XG4gICAgICAgIGNvbnN0IGluYm91bmROb2RlSW5kZXggPSBpbnB1dERhdGFbMV07XG4gICAgICAgIGNvbnN0IGluYm91bmRUZW5zb3JJbmRleCA9IGlucHV0RGF0YVsyXTtcblxuICAgICAgICBrd2FyZ3MgPSBpbnB1dERhdGFbM10gPT0gbnVsbCA/XG4gICAgICAgICAgICB7fSA6XG4gICAgICAgICAgICBpbnB1dERhdGFbM10gYXMgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0O1xuICAgICAgICBpZiAoIShpbmJvdW5kTGF5ZXJOYW1lIGluIGNyZWF0ZWRMYXllcnMpKSB7XG4gICAgICAgICAgYWRkVW5wcm9jZXNzZWROb2RlKGxheWVyLCBub2RlRGF0YSk7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGluYm91bmRMYXllciA9IGNyZWF0ZWRMYXllcnNbaW5ib3VuZExheWVyTmFtZV07XG4gICAgICAgIGlmIChpbmJvdW5kTGF5ZXIuaW5ib3VuZE5vZGVzLmxlbmd0aCA8PSBpbmJvdW5kTm9kZUluZGV4KSB7XG4gICAgICAgICAgYWRkVW5wcm9jZXNzZWROb2RlKGxheWVyLCBub2RlRGF0YSk7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGluYm91bmROb2RlID0gaW5ib3VuZExheWVyLmluYm91bmROb2Rlc1tpbmJvdW5kTm9kZUluZGV4XTtcbiAgICAgICAgaW5wdXRUZW5zb3JzLnB1c2goaW5ib3VuZE5vZGUub3V0cHV0VGVuc29yc1tpbmJvdW5kVGVuc29ySW5kZXhdKTtcbiAgICAgIH1cbiAgICAgIC8vIENhbGwgbGF5ZXIgb24gaXRzIGlucHV0cywgdGh1cyBjcmVhdGluZyB0aGUgbm9kZVxuICAgICAgLy8gYW5kIGJ1aWxkaW5nIHRoZSBsYXllciBpZiBuZWVkZWQuXG4gICAgICAvLyBOb3RlOiBUaGlzIGhhcyBFYWdlciB2cyBHcmFwaCBJbXBsaWNhdGlvbnMuXG4gICAgICBpZiAoaW5wdXRUZW5zb3JzLmxlbmd0aCA+IDApIHtcbiAgICAgICAgbGF5ZXIuYXBwbHkoXG4gICAgICAgICAgICBnZW5lcmljX3V0aWxzLnNpbmdsZXRvbk9yQXJyYXkoaW5wdXRUZW5zb3JzKSxcbiAgICAgICAgICAgIGt3YXJncyk7ICAvLyB3YXMgKioga3dhcmdzXG4gICAgICB9XG4gICAgfVxuXG4gICAgLyoqXG4gICAgICogRGVzZXJpYWxpemUgYSBsYXllciwgdGhlbiBjYWxsIGl0IG9uIGFwcHJvcHJpYXRlIGlucHV0cy5cbiAgICAgKiBAcGFyYW0gbGF5ZXJEYXRhOiBsYXllciBjb25maWcgZGljdC5cbiAgICAgKiBAdGhyb3dzIFZhbHVlRXJyb3I6IEluIGNhc2Ugb2YgaW1wcm9wZXJseSBmb3JtYXR0ZWQgYGxheWVyX2RhdGFgXG4gICAgICogZGljdC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBwcm9jZXNzTGF5ZXIobGF5ZXJEYXRhOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3R8bnVsbCkge1xuICAgICAgY29uc3QgbGF5ZXJOYW1lID0gbGF5ZXJEYXRhWyduYW1lJ10gYXMgc3RyaW5nO1xuICAgICAgLy8gSW5zdGFudGlhdGUgbGF5ZXIuXG4gICAgICBjb25zdCBsYXllciA9XG4gICAgICAgICAgZGVzZXJpYWxpemVMYXllcihcbiAgICAgICAgICAgICAgbGF5ZXJEYXRhLFxuICAgICAgICAgICAgICBjb25maWdbJ2N1c3RvbU9iamVjdHMnXSAhPSBudWxsID9cbiAgICAgICAgICAgICAgICAgIGNvbmZpZ1snY3VzdG9tT2JqZWN0cyddIGFzIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA6XG4gICAgICAgICAgICAgICAgICB7fSkgYXMgTGF5ZXI7XG4gICAgICBsYXllci5zZXRGYXN0V2VpZ2h0SW5pdER1cmluZ0J1aWxkKGZhc3RXZWlnaHRJbml0KTtcbiAgICAgIGNyZWF0ZWRMYXllcnNbbGF5ZXJOYW1lXSA9IGxheWVyO1xuICAgICAgLy8gR2F0aGVyIGxheWVyIGlucHV0cy5cbiAgICAgIGNvbnN0IGluYm91bmROb2Rlc0RhdGEgPVxuICAgICAgICAgIGxheWVyRGF0YVsnaW5ib3VuZE5vZGVzJ10gYXMgVGVuc29yS2V5V2l0aEFyZ3NBcnJheVtdW107XG4gICAgICBpbmJvdW5kTm9kZXNEYXRhLmZvckVhY2gobm9kZURhdGEgPT4ge1xuICAgICAgICBpZiAoIShub2RlRGF0YSBpbnN0YW5jZW9mIEFycmF5KSkge1xuICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgICBgQ29ycnVwdGVkIGNvbmZpZ3VyYXRpb24sIGV4cGVjdGVkIGFycmF5IGZvciBub2RlRGF0YTogJHtcbiAgICAgICAgICAgICAgICAgIG5vZGVEYXRhfWApO1xuICAgICAgICB9XG4gICAgICAgIC8vIFdlIGRvbid0IHByb2Nlc3Mgbm9kZXMgKGkuZS4gbWFrZSBsYXllciBjYWxscylcbiAgICAgICAgLy8gb24gdGhlIGZseSBiZWNhdXNlIHRoZSBpbmJvdW5kIG5vZGUgbWF5IG5vdCB5ZXQgZXhpc3QsXG4gICAgICAgIC8vIGluIGNhc2Ugb2YgbGF5ZXIgc2hhcmVkIGF0IGRpZmZlcmVudCB0b3BvbG9naWNhbCBkZXB0aHNcbiAgICAgICAgLy8gKGUuZy5hIG1vZGVsIHN1Y2ggYXMgQShCKEEoQih4KSkpKSlcbiAgICAgICAgYWRkVW5wcm9jZXNzZWROb2RlKGxheWVyLCBub2RlRGF0YSk7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICAvLyBGaXJzdCwgd2UgY3JlYXRlIGFsbCBsYXllcnMgYW5kIGVucXVldWUgbm9kZXMgdG8gYmUgcHJvY2Vzc2VkLlxuICAgIGNvbnN0IG5hbWUgPSBjb25maWdbJ25hbWUnXTtcbiAgICBjb25zdCBsYXllcnNGcm9tQ29uZmlnID0gY29uZmlnWydsYXllcnMnXSBhcyBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3RbXTtcbiAgICBmb3IgKGNvbnN0IGxheWVyRGF0YSBvZiBsYXllcnNGcm9tQ29uZmlnKSB7XG4gICAgICBwcm9jZXNzTGF5ZXIobGF5ZXJEYXRhKTtcbiAgICB9XG5cbiAgICAvLyBUaGVuIHdlIHByb2Nlc3Mgbm9kZXMgaW4gb3JkZXIgb2YgbGF5ZXIgZGVwdGguXG4gICAgLy8gTm9kZXMgdGhhdCBjYW5ub3QgeWV0IGJlIHByb2Nlc3NlZChpZiB0aGUgaW5ib3VuZCBub2RlXG4gICAgLy8gZG9lcyBub3QgeWV0IGV4aXN0KSBhcmUgcmUgLSBlbnF1ZXVlZCwgYW5kIHRoZSBwcm9jZXNzXG4gICAgLy8gaXMgcmVwZWF0ZWQgdW50aWwgYWxsIG5vZGVzIGFyZSBwcm9jZXNzZWQuXG4gICAgd2hpbGUgKCFnZW5lcmljX3V0aWxzLmlzT2JqZWN0RW1wdHkodW5wcm9jZXNzZWROb2RlcykpIHtcbiAgICAgIGZvciAoY29uc3QgbGF5ZXJEYXRhIG9mIGxheWVyc0Zyb21Db25maWcpIHtcbiAgICAgICAgY29uc3QgbGF5ZXIgPSBjcmVhdGVkTGF5ZXJzW2xheWVyRGF0YVsnbmFtZSddIGFzIHN0cmluZ107XG4gICAgICAgIGlmIChsYXllci5uYW1lIGluIHVucHJvY2Vzc2VkTm9kZXMpIHtcbiAgICAgICAgICBjb25zdCBjdXJyZW50VW5wcm9jZXNzZWROb2Rlc0ZvckxheWVyID0gdW5wcm9jZXNzZWROb2Rlc1tsYXllci5uYW1lXTtcbiAgICAgICAgICBkZWxldGUgdW5wcm9jZXNzZWROb2Rlc1tsYXllci5uYW1lXTtcbiAgICAgICAgICBmb3IgKGNvbnN0IG5vZGVEYXRhIG9mIGN1cnJlbnRVbnByb2Nlc3NlZE5vZGVzRm9yTGF5ZXIpIHtcbiAgICAgICAgICAgIHByb2Nlc3NOb2RlKGxheWVyLCBub2RlRGF0YSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgY29uc3QgaW5wdXRUZW5zb3JzOiBTeW1ib2xpY1RlbnNvcltdID0gW107XG4gICAgY29uc3Qgb3V0cHV0VGVuc29yczogU3ltYm9saWNUZW5zb3JbXSA9IFtdO1xuICAgIGNvbnN0IGlucHV0TGF5ZXJzRnJvbUNvbmZpZyA9XG4gICAgICAgIGNvbmZpZ1snaW5wdXRMYXllcnMnXSBhcyBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3RbXTtcbiAgICBmb3IgKGNvbnN0IGxheWVyRGF0YSBvZiBpbnB1dExheWVyc0Zyb21Db25maWcpIHtcbiAgICAgIGNvbnN0IGxheWVyTmFtZSA9IGxheWVyRGF0YVswXSBhcyBzdHJpbmc7XG4gICAgICBjb25zdCBub2RlSW5kZXggPSBsYXllckRhdGFbMV0gYXMgbnVtYmVyO1xuICAgICAgY29uc3QgdGVuc29ySW5kZXggPSBsYXllckRhdGFbMl0gYXMgbnVtYmVyO1xuICAgICAgZ2VuZXJpY191dGlscy5hc3NlcnQobGF5ZXJOYW1lIGluIGNyZWF0ZWRMYXllcnMpO1xuICAgICAgY29uc3QgbGF5ZXIgPSBjcmVhdGVkTGF5ZXJzW2xheWVyTmFtZV07XG4gICAgICBjb25zdCBsYXllck91dHB1dFRlbnNvcnMgPSBsYXllci5pbmJvdW5kTm9kZXNbbm9kZUluZGV4XS5vdXRwdXRUZW5zb3JzO1xuICAgICAgaW5wdXRUZW5zb3JzLnB1c2gobGF5ZXJPdXRwdXRUZW5zb3JzW3RlbnNvckluZGV4XSk7XG4gICAgfVxuICAgIGNvbnN0IG91dHB1dExheWVyc0Zyb21Db25maWcgPVxuICAgICAgICBjb25maWdbJ291dHB1dExheWVycyddIGFzIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdFtdO1xuICAgIGZvciAoY29uc3QgbGF5ZXJEYXRhIG9mIG91dHB1dExheWVyc0Zyb21Db25maWcpIHtcbiAgICAgIGNvbnN0IGxheWVyTmFtZSA9IGxheWVyRGF0YVswXSBhcyBzdHJpbmc7XG4gICAgICBjb25zdCBub2RlSW5kZXggPSBsYXllckRhdGFbMV0gYXMgbnVtYmVyO1xuICAgICAgY29uc3QgdGVuc29ySW5kZXggPSBsYXllckRhdGFbMl0gYXMgbnVtYmVyO1xuICAgICAgZ2VuZXJpY191dGlscy5hc3NlcnQobGF5ZXJOYW1lIGluIGNyZWF0ZWRMYXllcnMpO1xuICAgICAgY29uc3QgbGF5ZXIgPSBjcmVhdGVkTGF5ZXJzW2xheWVyTmFtZV07XG4gICAgICBjb25zdCBsYXllck91dHB1dFRlbnNvcnMgPSBsYXllci5pbmJvdW5kTm9kZXNbbm9kZUluZGV4XS5vdXRwdXRUZW5zb3JzO1xuICAgICAgb3V0cHV0VGVuc29ycy5wdXNoKGxheWVyT3V0cHV0VGVuc29yc1t0ZW5zb3JJbmRleF0pO1xuICAgIH1cbiAgICByZXR1cm4gbmV3IGNscyh7aW5wdXRzOiBpbnB1dFRlbnNvcnMsIG91dHB1dHM6IG91dHB1dFRlbnNvcnMsIG5hbWV9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBEZXRlcm1pbmUgd2hldGhlciB0aGUgY29udGFpbmVyIGlzIHN0YXRlZnVsLlxuICAgKlxuICAgKiBQb3J0aW5nIE5vdGU6IHRoaXMgaXMgdGhlIGVxdWl2YWxlbnQgb2YgdGhlIHN0YXRlZnVsIEBwcm9wZXJ0eSBvZlxuICAgKiAgIHRoZSBDb250YWluZXIgY2xhc3MgaW4gUHlLZXJhcy5cbiAgICovXG4gIG92ZXJyaWRlIGdldCBzdGF0ZWZ1bCgpOiBib29sZWFuIHtcbiAgICAvLyBQb3J0aW5nIE5vdGU6IFRoaXMgY2hlY2sgaXMgdG8gcHJldmVudCBpbmFkdmVydGVudCBzZXR0aW5nIG9mIHRoZVxuICAgIC8vICAgX3N0YXRlZnVsIHByb3BlcnR5IG9mIHRoZSBDb250YWluZXIgaW5zdGFuY2UuXG4gICAgaWYgKHRoaXMuX3N0YXRlZnVsKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAnQ29udGFpbmVyIGluc3RhbmNlIHVuZXhwZWN0ZWRseSBoYXMgX3N0YXRlZnVsID0gdHJ1ZS4gVGhlICcgK1xuICAgICAgICAgICdzdGF0ZWZ1bG5lc3Mgb2YgYSBDb250YWluZXIgaXMgZGV0ZXJtaW5lZCBieSB0aGUgTGF5ZXJzIGl0ICcgK1xuICAgICAgICAgICdjb250YWlucy4gSXRzIF9zdGF0ZWZ1bCBwcm9wZXJ0eSBtdXN0IHJlbWFpbiB0aGUgZGVmYXVsdCBmYWxzZS4nKTtcbiAgICB9XG4gICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLmxheWVycykge1xuICAgICAgaWYgKGxheWVyLnN0YXRlZnVsKSB7XG4gICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cblxuICAvKipcbiAgICogUmVzZXQgdGhlIHN0YXRlIG9mIGFsbCBzdGF0ZWZ1bCBjb25zdGl0dWVudCBsYXllcnMgKGlmIGFueSkuXG4gICAqXG4gICAqIEV4YW1wbGVzIG9mIHN0YXRlZnVsIGxheWVycyBpbmNsdWRlIFJOTiBsYXllcnMgd2hvc2UgYHN0YXRlZnVsYCBwcm9wZXJ0eVxuICAgKiBpcyBzZXQgYXMgYHRydWVgLlxuICAgKi9cbiAgb3ZlcnJpZGUgcmVzZXRTdGF0ZXMoKSB7XG4gICAgdGlkeSgoKSA9PiB7XG4gICAgICB0aGlzLmxheWVycy5mb3JFYWNoKGxheWVyID0+IHtcbiAgICAgICAgLy8gdHNsaW50OmRpc2FibGU6bm8tYW55XG4gICAgICAgIGlmIChsYXllci5zdGF0ZWZ1bCkge1xuICAgICAgICAgIGxheWVyLnJlc2V0U3RhdGVzKCk7XG4gICAgICAgIH1cbiAgICAgICAgLy8gdHNsaW50OmVuYWJsZTpuby1hbnlcbiAgICAgIH0pO1xuICAgIH0pO1xuICB9XG59XG4iXX0=