/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { env, keep, tidy, util } from '@tensorflow/tfjs-core';
import { getNodeNameAndIndex, getParamValue, getTensor, getTensorsForCurrentContenxt, parseNodeName } from '../operations/executors/utils';
import { executeOp } from '../operations/operation_executor';
import { ExecutionContext } from './execution_context';
import { getExecutionSubgraph, getNodesInTopologicalOrder, isControlFlow } from './model_analysis';
export class GraphExecutor {
    /**
     *
     * @param graph Graph the model or function graph to be executed.
     * @param parent When building function exector you need to set the parent
     * executor. Since the weights and function executor maps are set at parant
     * level, that function executor can access the function maps and weight maps
     * through the parent.
     */
    constructor(graph, parent) {
        this.graph = graph;
        this.parent = parent;
        this.compiledMap = new Map();
        this._weightMap = {};
        this.SEPERATOR = ',';
        this._functions = {};
        this._functionExecutorMap = {};
        this.keepIntermediateTensors = false;
        this._outputs = graph.outputs;
        this._inputs = graph.inputs;
        this._initNodes = graph.initNodes;
        this._signature = graph.signature;
        this._functions = graph.functions;
        // create sub-graph executors
        if (graph.functions != null) {
            Object.keys(graph.functions).forEach(name => {
                this._functionExecutorMap[name] =
                    new GraphExecutor(graph.functions[name], this);
            });
        }
    }
    get weightIds() {
        return this.parent ? this.parent.weightIds : this._weightIds;
    }
    get functionExecutorMap() {
        return this.parent ? this.parent.functionExecutorMap :
            this._functionExecutorMap;
    }
    get weightMap() {
        return this.parent ? this.parent.weightMap : this._weightMap;
    }
    set weightMap(weightMap) {
        const weightIds = Object.keys(weightMap).map(key => weightMap[key].map(tensor => tensor.id));
        this._weightIds = [].concat(...weightIds);
        this._weightMap = weightMap;
    }
    /**
     * Set `ResourceManager` shared by executors of a model.
     * @param resourceManager: `ResourceManager` of the `GraphModel`.
     */
    set resourceManager(resourceManager) {
        this._resourceManager = resourceManager;
    }
    get inputs() {
        return this._inputs.map(node => {
            return {
                name: node.name,
                shape: node.attrParams['shape'] ?
                    node.attrParams['shape'].value :
                    undefined,
                dtype: node.attrParams['dtype'] ?
                    node.attrParams['dtype'].value :
                    undefined
            };
        });
    }
    get outputs() {
        return this._outputs.map(node => {
            return {
                name: node.name,
                shape: node.attrParams['shape'] ?
                    node.attrParams['shape'].value :
                    undefined,
                dtype: node.attrParams['dtype'] ?
                    node.attrParams['dtype'].value :
                    undefined
            };
        });
    }
    get inputNodes() {
        return this._inputs.map(node => node.signatureKey || node.name);
    }
    get outputNodes() {
        return this._outputs.map((node) => {
            const name = node.signatureKey || node.name;
            return node.defaultOutput ? (`${name}:${node.defaultOutput}`) : name;
        });
    }
    get functions() {
        return Object.keys(this._functions).reduce((map, key) => {
            map[key] = this._functions[key].signature;
            return map;
        }, {});
    }
    getCompilationKey(inputs, outputs) {
        const sortedInputs = inputs.map(node => node.name).sort();
        const sortedOutputs = outputs.map(node => node.name).sort();
        return sortedInputs.join(this.SEPERATOR) + '--' +
            sortedOutputs.join(this.SEPERATOR);
    }
    /**
     * Compiles the inference graph and returns the minimal set of nodes that are
     * required for execution, in the correct execution order.
     */
    compile(inputs, outputs) {
        const executionInfo = getExecutionSubgraph(inputs, outputs, this.weightMap, this._initNodes);
        const { missingInputs, dynamicNode, syncInputs } = executionInfo;
        if (dynamicNode != null) {
            throw new Error(`This execution contains the node '${dynamicNode.name}', which has ` +
                `the dynamic op '${dynamicNode.op}'. Please use ` +
                `model.executeAsync() instead. Alternatively, to avoid the ` +
                `dynamic ops, specify the inputs [${syncInputs}]`);
        }
        if (missingInputs.length > 0) {
            const outNames = outputs.map(n => n.name);
            const inNames = Object.keys(inputs);
            throw new Error(`Cannot compute the outputs [${outNames}] from the provided inputs ` +
                `[${inNames}]. Missing the following inputs: [${missingInputs}]`);
        }
        return getNodesInTopologicalOrder(this.graph, this.weightMap, executionInfo);
    }
    cloneAndKeepTensor(tensor) {
        if (tensor == null) {
            return null;
        }
        const clone = tensor.clone();
        // Keep the clone because`model.execute()` may be called within
        // a `tidy()`, but the user may inspect these tensors after the
        // tidy.
        keep(clone);
        return clone;
    }
    cloneTensorList(tensors) {
        if (!tensors) {
            return null;
        }
        const clonedTensor = tensors.map(tensor => {
            return this.cloneAndKeepTensor(tensor);
        });
        return clonedTensor;
    }
    cloneTensorMap(tensorsMap) {
        return Object.fromEntries(Object.entries(tensorsMap).map(([name, tensorsList]) => {
            return [name, this.cloneTensorList(tensorsList)];
        }));
    }
    /**
     * Executes the inference for given input tensors.
     * @param inputs Tensor map for the model inputs, keyed by the input node
     * names.
     * @param outputs Optional. output node name from the Tensorflow model, if
     * no outputs are specified, the default outputs of the model would be used.
     * You can inspect intermediate nodes of the model by adding them to the
     * outputs array.
     */
    execute(inputs, outputs) {
        // Dispose any tensors from a prior run to avoid leaking them.
        this.disposeIntermediateTensors();
        inputs = this.mapInputs(inputs);
        const names = Object.keys(inputs).sort();
        this.checkInputs(inputs);
        this.checkInputShapeAndType(inputs);
        outputs = this.mapOutputs(outputs);
        this.checkOutputs(outputs);
        const inputNodes = names.map(name => this.graph.nodes[parseNodeName(name)[0]]);
        const outputNodeNames = outputs.map(name => parseNodeName(name)[0]);
        let outputNodes = outputNodeNames.map(name => this.graph.nodes[name]);
        // If no outputs are specified, then use the default outputs of the model.
        if (outputNodes.length === 0) {
            outputNodes = this._outputs;
        }
        const compilationKey = this.getCompilationKey(inputNodes, outputNodes);
        // Do nothing if the compiled graph cache contains the input.
        let orderedNodes = this.compiledMap.get(compilationKey);
        if (orderedNodes == null) {
            orderedNodes = this.compile(inputs, outputNodes);
            this.compiledMap.set(compilationKey, orderedNodes);
        }
        // Keep tensors if KEEP_INTERMEDIATE_TENSORS is on.
        try {
            this.keepIntermediateTensors = env().getBool('KEEP_INTERMEDIATE_TENSORS');
        }
        catch (e) {
            this.keepIntermediateTensors = false;
            console.warn(e.message);
        }
        const tensorArrayMap = {};
        const tensorListMap = {};
        return tidy(() => {
            const context = new ExecutionContext(this.weightMap, tensorArrayMap, tensorListMap, this.functionExecutorMap);
            const tensorsMap = Object.assign({}, this.weightMap);
            if (this.keepIntermediateTensors) {
                this.clonedTensorsMap = this.cloneTensorMap(this.weightMap);
            }
            Object.keys(inputs).forEach(name => {
                const [nodeName, index] = parseNodeName(name);
                const tensors = [];
                tensors[index] = inputs[name];
                tensorsMap[nodeName] = tensors;
                if (this.keepIntermediateTensors) {
                    this.clonedTensorsMap[nodeName] = this.cloneTensorList(tensors);
                }
            });
            const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
            const intermediateTensorConsumerCount = {};
            for (let i = 0; i < orderedNodes.length; i++) {
                const node = orderedNodes[i];
                if (!tensorsMap[node.name]) {
                    const tensors = executeOp(node, tensorsMap, context, this._resourceManager);
                    if (util.isPromise(tensors)) {
                        throw new Error(`The execution of the op '${node.op}' returned a promise. ` +
                            `Please use model.executeAsync() instead.`);
                    }
                    tensorsMap[node.name] = tensors;
                    if (this.keepIntermediateTensors) {
                        this.clonedTensorsMap[node.name] = this.cloneTensorList(tensors);
                    }
                    this.checkTensorForDisposal(node.name, node, tensorsMap, context, tensorsToKeep, outputNodeNames, intermediateTensorConsumerCount);
                }
            }
            // dispose the context for the root executor
            if (this.parent == null) {
                context.dispose(tensorsToKeep);
            }
            return outputs.map(name => getTensor(name, tensorsMap, context));
        });
    }
    getFrozenTensorIds(tensorMap) {
        const ids = [].concat.apply([], Object.keys(tensorMap)
            .map(key => tensorMap[key])
            .map(tensors => tensors.map(tensor => tensor.id)));
        return new Set(ids);
    }
    checkTensorForDisposal(nodeName, node, tensorMap, context, tensorsToKeep, outputNames, intermediateTensorConsumerCount) {
        // Skip output nodes and any control flow nodes, since its dependency is
        // tricky to track correctly.
        if (node.category === 'control' || outputNames.indexOf(nodeName) !== -1) {
            return;
        }
        tensorMap[nodeName].forEach(tensor => {
            if (tensor != null) {
                intermediateTensorConsumerCount[tensor.id] =
                    (intermediateTensorConsumerCount[tensor.id] || 0) +
                        node.children.length;
            }
        });
        node.inputs.forEach(input => {
            // Skip any control flow nodes, since its dependency is tricky to track
            // correctly.
            if (input.category !== 'control') {
                const tensors = getTensorsForCurrentContenxt(input.name, tensorMap, context);
                if (tensors != null) {
                    tensors.forEach(tensor => {
                        if (tensor && !tensor.kept && !tensorsToKeep.has(tensor.id)) {
                            const count = intermediateTensorConsumerCount[tensor.id];
                            if (count === 1) {
                                tensor.dispose();
                                delete intermediateTensorConsumerCount[tensor.id];
                            }
                            else if (count != null) {
                                // only intermediate nodes has count set, inputs and weights
                                // are not.
                                intermediateTensorConsumerCount[tensor.id]--;
                            }
                        }
                    });
                }
            }
        });
    }
    /**
     * Executes the inference for given input tensors in Async fashion.
     * @param inputs Tensor map for the model inputs, keyed by the input node
     * names.
     * @param outputs output node name from the Tensorflow model, if no outputs
     * are specified, the default outputs of the model would be used. You can
     * inspect intermediate nodes of the model by adding them to the outputs
     * array.
     */
    async executeAsync(inputs, outputs) {
        return this._executeAsync(inputs, outputs);
    }
    disposeIntermediateTensors() {
        if (!this.clonedTensorsMap) {
            return;
        }
        Object.values(this.clonedTensorsMap).forEach(tensorsList => {
            for (const tensor of tensorsList) {
                if (tensor && !tensor.isDisposed) {
                    tensor.dispose();
                }
            }
        });
        this.clonedTensorsMap = null;
    }
    getIntermediateTensors() {
        return this.clonedTensorsMap;
    }
    /**
     * Executes the inference for given input tensors in Async fashion.
     * @param inputs Tensor map for the model inputs, keyed by the input node
     * names.
     * @param outputs Optional. output node name from the Tensorflow model,
     * if no outputs are specified, the default outputs of the model would be
     * used. You can inspect intermediate nodes of the model by adding them to
     * the outputs array.
     * @param isFunctionExecution Optional. Flag for executing a function.
     * @param tensorArrayMap Optional, global TensorArray map by id. Used for
     * function execution.
     * @param tensorArrayMap Optinal global TensorList map by id. Used for
     * function execution.
     */
    async _executeAsync(inputs, outputs, isFunctionExecution = false, tensorArrayMap = {}, tensorListMap = {}) {
        // Dispose any tensors from a prior run to avoid leaking them.
        this.disposeIntermediateTensors();
        if (!isFunctionExecution) {
            inputs = this.mapInputs(inputs);
            this.checkInputs(inputs);
            this.checkInputShapeAndType(inputs);
            outputs = this.mapOutputs(outputs);
            this.checkOutputs(outputs);
        }
        // Keep tensors if KEEP_INTERMEDIATE_TENSORS is on.
        try {
            this.keepIntermediateTensors = env().getBool('KEEP_INTERMEDIATE_TENSORS');
        }
        catch (e) {
            this.keepIntermediateTensors = false;
            console.warn(e.message);
        }
        const context = new ExecutionContext(this.weightMap, tensorArrayMap, tensorListMap, this.functionExecutorMap);
        if (this.keepIntermediateTensors) {
            this.clonedTensorsMap = this.cloneTensorMap(this.weightMap);
        }
        // Graph with control flow op requires runtime evaluation of the execution
        // order, while without control flow the execution order is pre-determined
        // in the compile method.
        const tensorsMap = await this.executeWithControlFlow(inputs, context, outputs, isFunctionExecution);
        const results = outputs.map(name => getTensor(name, tensorsMap, context));
        // dispose all the intermediate tensors
        const outputIds = results.map(t => t.id);
        const inputIds = Object.keys(inputs).map(name => inputs[name].id);
        const keepIds = new Set([...outputIds, ...inputIds, ...this.weightIds]);
        Object.values(tensorsMap).forEach(tensorsList => {
            tensorsList.forEach(tensor => {
                if (tensor && !tensor.isDisposed && !keepIds.has(tensor.id)) {
                    tensor.dispose();
                }
            });
        });
        // dispose the context for the root executor
        if (this.parent == null) {
            context.dispose(keepIds);
        }
        return results;
    }
    async executeFunctionAsync(inputs, tensorArrayMap, tensorListMap) {
        const mappedInputs = inputs.reduce((map, tensor, index) => {
            map[this.inputs[index].name] = tensor;
            return map;
        }, {});
        return this._executeAsync(mappedInputs, this.outputNodes, true, tensorArrayMap, tensorListMap);
    }
    /**
     * When there are control flow nodes in the graph, the graph execution use
     * ExecutionContext to keep track of the frames and loop iterators.
     * @param inputs placeholder tensors for the graph.
     * @param context the execution context object for current execution.
     * @param outputNames Optional. output node name from the Tensorflow model,
     * if no outputs are specified, the default outputs of the model would be
     * used. You can inspect intermediate nodes of the model by adding them to
     * the outputs array.
     * @param isFunctionExecution Flag for executing a function.
     */
    async executeWithControlFlow(inputs, context, outputNames, isFunctionExecution) {
        const names = Object.keys(inputs);
        const inputNodes = names.map(name => this.graph.nodes[parseNodeName(name)[0]]);
        const outputNodeNames = outputNames.map(name => parseNodeName(name)[0]);
        let outputNodes = outputNodeNames.map(name => this.graph.nodes[name]);
        // If no outputs are specified, then use the default outputs of the model.
        if (outputNodes.length === 0) {
            outputNodes = this._outputs;
        }
        const { usedNodes, missingInputs, dynamicNode, syncInputs } = getExecutionSubgraph(inputs, outputNodes, this.weightMap, this._initNodes);
        // First nodes to execute include inputNodes, weights, and initNodes.
        const stack = [
            ...inputNodes, ...this.graph.weights, ...(this._initNodes || [])
        ].map(node => {
            return { node, contexts: context.currentContext };
        });
        const tensorsMap = Object.assign({}, this.weightMap);
        Object.keys(inputs).forEach(name => {
            const [nodeName, index] = parseNodeName(name);
            const tensors = [];
            tensors[index] = inputs[name];
            tensorsMap[nodeName] = tensors;
        });
        const intermediateTensorConsumerCount = {};
        const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
        const added = {};
        while (stack.length > 0) {
            const promises = this.processStack(inputNodes, stack, context, tensorsMap, added, tensorsToKeep, outputNodeNames, intermediateTensorConsumerCount, usedNodes);
            await Promise.all(promises);
        }
        if (dynamicNode == null && !isFunctionExecution) {
            console.warn(`This model execution did not contain any nodes with control flow ` +
                `or dynamic output shapes. You can use model.execute() instead.`);
        }
        const missingOutputs = outputNodes
            .filter(node => !isControlFlow(node) &&
            !getTensor(node.name, tensorsMap, context))
            .map(node => node.name);
        if (missingOutputs.length > 0) {
            let alternativeMsg = '';
            if (dynamicNode != null) {
                alternativeMsg =
                    `Alternatively, to avoid the dynamic ops, use model.execute() ` +
                        `and specify the inputs [${syncInputs}]`;
            }
            throw new Error(`Cannot compute the outputs [${missingOutputs}] from the provided ` +
                `inputs [${names}]. Consider providing the following inputs: ` +
                `[${missingInputs}]. ${alternativeMsg}`);
        }
        return tensorsMap;
    }
    processStack(inputNodes, stack, context, tensorMap, added, tensorsToKeep, outputNames, intermediateTensorConsumerCount, usedNodes) {
        const promises = [];
        while (stack.length > 0) {
            const item = stack.pop();
            context.currentContext = item.contexts;
            let nodeName = '';
            // The tensor of the Enter op with isConstant set should be set
            // in the parent scope, so it will be available as constant for the
            // whole loop.
            if (item.node.op === 'Enter' &&
                getParamValue('isConstant', item.node, tensorMap, context)) {
                [nodeName] = getNodeNameAndIndex(item.node.name, context);
            }
            // only process nodes that are not in the tensorMap yet, this include
            // inputNodes and internal initNodes.
            if (tensorMap[item.node.name] == null) {
                const tensors = executeOp(item.node, tensorMap, context, this._resourceManager);
                if (!nodeName) {
                    [nodeName] = getNodeNameAndIndex(item.node.name, context);
                }
                const currentContext = context.currentContext;
                if (util.isPromise(tensors)) {
                    promises.push(tensors.then(t => {
                        tensorMap[nodeName] = t;
                        if (this.keepIntermediateTensors) {
                            this.clonedTensorsMap[nodeName] = this.cloneTensorList(t);
                        }
                        context.currentContext = currentContext;
                        this.checkTensorForDisposal(nodeName, item.node, tensorMap, context, tensorsToKeep, outputNames, intermediateTensorConsumerCount);
                        this.processChildNodes(item.node, stack, context, tensorMap, added, usedNodes);
                        return t;
                    }));
                }
                else {
                    tensorMap[nodeName] = tensors;
                    if (this.keepIntermediateTensors) {
                        this.clonedTensorsMap[nodeName] = this.cloneTensorList(tensors);
                    }
                    this.checkTensorForDisposal(nodeName, item.node, tensorMap, context, tensorsToKeep, outputNames, intermediateTensorConsumerCount);
                    this.processChildNodes(item.node, stack, context, tensorMap, added, usedNodes);
                }
            }
            else {
                this.processChildNodes(item.node, stack, context, tensorMap, added, usedNodes);
            }
        }
        return promises;
    }
    processChildNodes(node, stack, context, tensorMap, added, usedNodes) {
        node.children.forEach((childNode) => {
            const [nodeName,] = getNodeNameAndIndex(childNode.name, context);
            if (added[nodeName] || !usedNodes.has(childNode.name)) {
                return;
            }
            // Merge op can be pushed if any of its inputs has value.
            if (childNode.op === 'Merge') {
                if (childNode.inputNames.some(name => {
                    return !!getTensor(name, tensorMap, context);
                })) {
                    added[nodeName] = true;
                    stack.push({ contexts: context.currentContext, node: childNode });
                }
            }
            else // Otherwise all inputs must to have value.
             if (childNode.inputNames.every(name => {
                return !!getTensor(name, tensorMap, context);
            })) {
                added[nodeName] = true;
                stack.push({ contexts: context.currentContext, node: childNode });
            }
        });
    }
    /**
     * Releases the memory used by the weight tensors.
     */
    dispose() {
        Object.keys(this.weightMap)
            .forEach(key => this.weightMap[key].forEach(tensor => tensor.dispose()));
    }
    checkInputShapeAndType(inputs) {
        Object.keys(inputs).forEach(name => {
            const input = inputs[name];
            const [nodeName,] = parseNodeName(name);
            const node = this.graph.nodes[nodeName];
            if (node.attrParams['shape'] && node.attrParams['shape'].value) {
                const shape = node.attrParams['shape'].value;
                const match = shape.length === input.shape.length &&
                    input.shape.every((dim, index) => shape[index] === -1 || shape[index] === dim);
                util.assert(match, () => `The shape of dict['${node.name}'] provided in ` +
                    `model.execute(dict) must be [${shape}], but was ` +
                    `[${input.shape}]`);
            }
            if (node.attrParams['dtype'] && node.attrParams['dtype'].value) {
                util.assert(input.dtype === node.attrParams['dtype'].value, () => `The dtype of dict['${node.name}'] provided in ` +
                    `model.execute(dict) must be ` +
                    `${node.attrParams['dtype'].value}, but was ${input.dtype}`);
            }
        });
    }
    mapInputs(inputs) {
        var _a, _b;
        const result = {};
        for (const inputName in inputs) {
            const tensor = (_b = (_a = this._signature) === null || _a === void 0 ? void 0 : _a.inputs) === null || _b === void 0 ? void 0 : _b[inputName];
            if (tensor != null) {
                result[tensor.name] = inputs[inputName];
            }
            else {
                result[inputName] = inputs[inputName];
            }
        }
        return result;
    }
    checkInputs(inputs) {
        const notInGraph = Object.keys(inputs).filter(name => {
            const [nodeName] = parseNodeName(name);
            return this.graph.nodes[nodeName] == null;
        });
        if (notInGraph.length > 0) {
            throw new Error(`The dict provided in model.execute(dict) has ` +
                `keys: [${notInGraph}] that are not part of graph`);
        }
    }
    mapOutputs(outputs) {
        return outputs.map(name => {
            var _a, _b;
            const tensor = (_b = (_a = this._signature) === null || _a === void 0 ? void 0 : _a.outputs) === null || _b === void 0 ? void 0 : _b[name];
            if (tensor != null) {
                return tensor.name;
            }
            return name;
        }, {});
    }
    checkOutputs(outputs) {
        outputs.forEach(name => {
            const [normalizedName] = parseNodeName(name);
            if (!this.graph.nodes[normalizedName]) {
                throw new Error(`The output '${name}' is not found in the graph`);
            }
        });
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3JhcGhfZXhlY3V0b3IuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvbnZlcnRlci9zcmMvZXhlY3V0b3IvZ3JhcGhfZXhlY3V0b3IudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFXLEdBQUcsRUFBRSxJQUFJLEVBQTBCLElBQUksRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUk5RixPQUFPLEVBQUMsbUJBQW1CLEVBQUUsYUFBYSxFQUFFLFNBQVMsRUFBRSw0QkFBNEIsRUFBRSxhQUFhLEVBQUMsTUFBTSwrQkFBK0IsQ0FBQztBQUN6SSxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sa0NBQWtDLENBQUM7QUFHM0QsT0FBTyxFQUFDLGdCQUFnQixFQUF1QixNQUFNLHFCQUFxQixDQUFDO0FBQzNFLE9BQU8sRUFBQyxvQkFBb0IsRUFBRSwwQkFBMEIsRUFBRSxhQUFhLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQVNqRyxNQUFNLE9BQU8sYUFBYTtJQXlGeEI7Ozs7Ozs7T0FPRztJQUNILFlBQW9CLEtBQVksRUFBVSxNQUFzQjtRQUE1QyxVQUFLLEdBQUwsS0FBSyxDQUFPO1FBQVUsV0FBTSxHQUFOLE1BQU0sQ0FBZ0I7UUFoR3hELGdCQUFXLEdBQXdCLElBQUksR0FBRyxFQUFFLENBQUM7UUFDN0MsZUFBVSxHQUFvQixFQUFFLENBQUM7UUFNakMsY0FBUyxHQUFHLEdBQUcsQ0FBQztRQUNoQixlQUFVLEdBQTJCLEVBQUUsQ0FBQztRQUN4Qyx5QkFBb0IsR0FBc0MsRUFBRSxDQUFDO1FBRzdELDRCQUF1QixHQUFHLEtBQUssQ0FBQztRQXFGdEMsSUFBSSxDQUFDLFFBQVEsR0FBRyxLQUFLLENBQUMsT0FBTyxDQUFDO1FBQzlCLElBQUksQ0FBQyxPQUFPLEdBQUcsS0FBSyxDQUFDLE1BQU0sQ0FBQztRQUM1QixJQUFJLENBQUMs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QUFlLElBQUksNkJBQTZCLENBQUMsQ0FBQzthQUNuRTtRQUNILENBQUMsQ0FBQyxDQUFDO0lBQ0wsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0RhdGFUeXBlLCBlbnYsIGtlZXAsIE5hbWVkVGVuc29yTWFwLCBUZW5zb3IsIHRpZHksIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7SVNpZ25hdHVyZURlZn0gZnJvbSAnLi4vZGF0YS9jb21waWxlZF9hcGknO1xuaW1wb3J0IHtOYW1lZFRlbnNvcnNNYXAsIFRlbnNvckFycmF5TWFwLCBUZW5zb3JJbmZvLCBUZW5zb3JMaXN0TWFwfSBmcm9tICcuLi9kYXRhL3R5cGVzJztcbmltcG9ydCB7Z2V0Tm9kZU5hbWVBbmRJbmRleCwgZ2V0UGFyYW1WYWx1ZSwgZ2V0VGVuc29yLCBnZXRUZW5zb3JzRm9yQ3VycmVudENvbnRlbnh0LCBwYXJzZU5vZGVOYW1lfSBmcm9tICcuLi9vcGVyYXRpb25zL2V4ZWN1dG9ycy91dGlscyc7XG5pbXBvcnQge2V4ZWN1dGVPcH0gZnJvbSAnLi4vb3BlcmF0aW9ucy9vcGVyYXRpb25fZXhlY3V0b3InO1xuaW1wb3J0IHtHcmFwaCwgTm9kZX0gZnJvbSAnLi4vb3BlcmF0aW9ucy90eXBlcyc7XG5cbmltcG9ydCB7RXhlY3V0aW9uQ29udGV4dCwgRXhlY3V0aW9uQ29udGV4dEluZm99IGZyb20gJy4vZXhlY3V0aW9uX2NvbnRleHQnO1xuaW1wb3J0IHtnZXRFeGVjdXRpb25TdWJncmFwaCwgZ2V0Tm9kZXNJblRvcG9sb2dpY2FsT3JkZXIsIGlzQ29udHJvbEZsb3d9IGZyb20gJy4vbW9kZWxfYW5hbHlzaXMnO1xuaW1wb3J0IHtSZXNvdXJjZU1hbmFnZXJ9IGZyb20gJy4vcmVzb3VyY2VfbWFuYWdlcic7XG5pbXBvcnQge0Z1bmN0aW9uRXhlY3V0b3J9IGZyb20gJy4vdHlwZXMnO1xuXG5pbnRlcmZhY2UgTm9kZVdpdGhDb250ZXh0cyB7XG4gIGNvbnRleHRzOiBFeGVjdXRpb25Db250ZXh0SW5mb1tdO1xuICBub2RlOiBOb2RlO1xufVxuXG5leHBvcnQgY2xhc3MgR3JhcGhFeGVjdXRvciBpbXBsZW1lbnRzIEZ1bmN0aW9uRXhlY3V0b3Ige1xuICBwcml2YXRlIGNvbXBpbGVkTWFwOiBNYXA8c3RyaW5nLCBOb2RlW10+ID0gbmV3IE1hcCgpO1xuICBwcml2YXRlIF93ZWlnaHRNYXA6IE5hbWVkVGVuc29yc01hcCA9IHt9O1xuICBwcml2YXRlIF93ZWlnaHRJZHM6IG51bWJlcltdO1xuICBwcml2YXRlIF9zaWduYXR1cmU6IElTaWduYXR1cmVEZWY7XG4gIHByaXZhdGUgX2lucHV0czogTm9kZVtdO1xuICBwcml2YXRlIF9vdXRwdXRzOiBOb2RlW107XG4gIHByaXZhdGUgX2luaXROb2RlczogTm9kZVtdOyAgLy8gSW50ZXJuYWwgaW5pdCBub2RlcyB0byBzdGFydCBpbml0aWFsaXphdGlvbi5cbiAgcHJpdmF0ZSBTRVBFUkFUT1IgPSAnLCc7XG4gIHByaXZhdGUgX2Z1bmN0aW9uczoge1trZXk6IHN0cmluZ106IEdyYXBofSA9IHt9O1xuICBwcml2YXRlIF9mdW5jdGlvbkV4ZWN1dG9yTWFwOiB7W2tleTogc3RyaW5nXTogRnVuY3Rpb25FeGVjdXRvcn0gPSB7fTtcbiAgcHJpdmF0ZSBfcmVzb3VyY2VNYW5hZ2VyOiBSZXNvdXJjZU1hbmFnZXI7XG4gIHByaXZhdGUgY2xvbmVkVGVuc29yc01hcDogTmFtZWRUZW5zb3JzTWFwO1xuICBwcml2YXRlIGtlZXBJbnRlcm1lZGlhdGVUZW5zb3JzID0gZmFsc2U7XG5cbiAgZ2V0IHdlaWdodElkcygpOiBudW1iZXJbXSB7XG4gICAgcmV0dXJuIHRoaXMucGFyZW50ID8gdGhpcy5wYXJlbnQud2VpZ2h0SWRzIDogdGhpcy5fd2VpZ2h0SWRzO1xuICB9XG5cbiAgZ2V0IGZ1bmN0aW9uRXhlY3V0b3JNYXAoKToge1trZXk6IHN0cmluZ106IEZ1bmN0aW9uRXhlY3V0b3J9IHtcbiAgICByZXR1cm4gdGhpcy5wYXJlbnQgPyB0aGlzLnBhcmVudC5mdW5jdGlvbkV4ZWN1dG9yTWFwIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLl9mdW5jdGlvbkV4ZWN1dG9yTWFwO1xuICB9XG5cbiAgZ2V0IHdlaWdodE1hcCgpOiBOYW1lZFRlbnNvcnNNYXAge1xuICAgIHJldHVybiB0aGlzLnBhcmVudCA/IHRoaXMucGFyZW50LndlaWdodE1hcCA6IHRoaXMuX3dlaWdodE1hcDtcbiAgfVxuXG4gIHNldCB3ZWlnaHRNYXAod2VpZ2h0TWFwOiBOYW1lZFRlbnNvcnNNYXApIHtcbiAgICBjb25zdCB3ZWlnaHRJZHMgPSBPYmplY3Qua2V5cyh3ZWlnaHRNYXApLm1hcChcbiAgICAgICAga2V5ID0+IHdlaWdodE1hcFtrZXldLm1hcCh0ZW5zb3IgPT4gdGVuc29yLmlkKSk7XG4gICAgdGhpcy5fd2VpZ2h0SWRzID0gW10uY29uY2F0KC4uLndlaWdodElkcyk7XG4gICAgdGhpcy5fd2VpZ2h0TWFwID0gd2VpZ2h0TWFwO1xuICB9XG5cbiAgLyoqXG4gICAqIFNldCBgUmVzb3VyY2VNYW5hZ2VyYCBzaGFyZWQgYnkgZXhlY3V0b3JzIG9mIGEgbW9kZWwuXG4gICAqIEBwYXJhbSByZXNvdXJjZU1hbmFnZXI6IGBSZXNvdXJjZU1hbmFnZXJgIG9mIHRoZSBgR3JhcGhNb2RlbGAuXG4gICAqL1xuICBzZXQgcmVzb3VyY2VNYW5hZ2VyKHJlc291cmNlTWFuYWdlcjogUmVzb3VyY2VNYW5hZ2VyKSB7XG4gICAgdGhpcy5fcmVzb3VyY2VNYW5hZ2VyID0gcmVzb3VyY2VNYW5hZ2VyO1xuICB9XG5cbiAgZ2V0IGlucHV0cygpOiBUZW5zb3JJbmZvW10ge1xuICAgIHJldHVybiB0aGlzLl9pbnB1dHMubWFwKG5vZGUgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgbmFtZTogbm9kZS5uYW1lLFxuICAgICAgICBzaGFwZTogbm9kZS5hdHRyUGFyYW1zWydzaGFwZSddID9cbiAgICAgICAgICAgIG5vZGUuYXR0clBhcmFtc1snc2hhcGUnXS52YWx1ZSBhcyBudW1iZXJbXSA6XG4gICAgICAgICAgICB1bmRlZmluZWQsXG4gICAgICAgIGR0eXBlOiBub2RlLmF0dHJQYXJhbXNbJ2R0eXBlJ10gP1xuICAgICAgICAgICAgbm9kZS5hdHRyUGFyYW1zWydkdHlwZSddLnZhbHVlIGFzIERhdGFUeXBlIDpcbiAgICAgICAgICAgIHVuZGVmaW5lZFxuICAgICAgfTtcbiAgICB9KTtcbiAgfVxuXG4gIGdldCBvdXRwdXRzKCk6IFRlbnNvckluZm9bXSB7XG4gICAgcmV0dXJuIHRoaXMuX291dHB1dHMubWFwKG5vZGUgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgbmFtZTogbm9kZS5uYW1lLFxuICAgICAgICBzaGFwZTogbm9kZS5hdHRyUGFyYW1zWydzaGFwZSddID9cbiAgICAgICAgICAgIG5vZGUuYXR0clBhcmFtc1snc2hhcGUnXS52YWx1ZSBhcyBudW1iZXJbXSA6XG4gICAgICAgICAgICB1bmRlZmluZWQsXG4gICAgICAgIGR0eXBlOiBub2RlLmF0dHJQYXJhbXNbJ2R0eXBlJ10gP1xuICAgICAgICAgICAgbm9kZS5hdHRyUGFyYW1zWydkdHlwZSddLnZhbHVlIGFzIERhdGFUeXBlIDpcbiAgICAgICAgICAgIHVuZGVmaW5lZFxuICAgICAgfTtcbiAgICB9KTtcbiAgfVxuXG4gIGdldCBpbnB1dE5vZGVzKCk6IHN0cmluZ1tdIHtcbiAgICByZXR1cm4gdGhpcy5faW5wdXRzLm1hcChub2RlID0+IG5vZGUuc2lnbmF0dXJlS2V5IHx8IG5vZGUubmFtZSk7XG4gIH1cblxuICBnZXQgb3V0cHV0Tm9kZXMoKTogc3RyaW5nW10ge1xuICAgIHJldHVybiB0aGlzLl9vdXRwdXRzLm1hcCgobm9kZSkgPT4ge1xuICAgICAgY29uc3QgbmFtZSA9IG5vZGUuc2lnbmF0dXJlS2V5IHx8IG5vZGUubmFtZTtcbiAgICAgIHJldHVybiBub2RlLmRlZmF1bHRPdXRwdXQgPyAoYCR7bmFtZX06JHtub2RlLmRlZmF1bHRPdXRwdXR9YCkgOiBuYW1lO1xuICAgIH0pO1xuICB9XG5cbiAgZ2V0IGZ1bmN0aW9ucygpOiB7W2tleTogc3RyaW5nXTogSVNpZ25hdHVyZURlZn0ge1xuICAgIHJldHVybiBPYmplY3Qua2V5cyh0aGlzLl9mdW5jdGlvbnMpLnJlZHVjZSgobWFwLCBrZXkpID0+IHtcbiAgICAgIG1hcFtrZXldID0gdGhpcy5fZnVuY3Rpb25zW2tleV0uc2lnbmF0dXJlO1xuICAgICAgcmV0dXJuIG1hcDtcbiAgICB9LCB7fSBhcyB7W2tleTogc3RyaW5nXTogSVNpZ25hdHVyZURlZn0pO1xuICB9XG5cbiAgLyoqXG4gICAqXG4gICAqIEBwYXJhbSBncmFwaCBHcmFwaCB0aGUgbW9kZWwgb3IgZnVuY3Rpb24gZ3JhcGggdG8gYmUgZXhlY3V0ZWQuXG4gICAqIEBwYXJhbSBwYXJlbnQgV2hlbiBidWlsZGluZyBmdW5jdGlvbiBleGVjdG9yIHlvdSBuZWVkIHRvIHNldCB0aGUgcGFyZW50XG4gICAqIGV4ZWN1dG9yLiBTaW5jZSB0aGUgd2VpZ2h0cyBhbmQgZnVuY3Rpb24gZXhlY3V0b3IgbWFwcyBhcmUgc2V0IGF0IHBhcmFudFxuICAgKiBsZXZlbCwgdGhhdCBmdW5jdGlvbiBleGVjdXRvciBjYW4gYWNjZXNzIHRoZSBmdW5jdGlvbiBtYXBzIGFuZCB3ZWlnaHQgbWFwc1xuICAgKiB0aHJvdWdoIHRoZSBwYXJlbnQuXG4gICAqL1xuICBjb25zdHJ1Y3Rvcihwcml2YXRlIGdyYXBoOiBHcmFwaCwgcHJpdmF0ZSBwYXJlbnQ/OiBHcmFwaEV4ZWN1dG9yKSB7XG4gICAgdGhpcy5fb3V0cHV0cyA9IGdyYXBoLm91dHB1dHM7XG4gICAgdGhpcy5faW5wdXRzID0gZ3JhcGguaW5wdXRzO1xuICAgIHRoaXMuX2luaXROb2RlcyA9IGdyYXBoLmluaXROb2RlcztcbiAgICB0aGlzLl9zaWduYXR1cmUgPSBncmFwaC5zaWduYXR1cmU7XG4gICAgdGhpcy5fZnVuY3Rpb25zID0gZ3JhcGguZnVuY3Rpb25zO1xuICAgIC8vIGNyZWF0ZSBzdWItZ3JhcGggZXhlY3V0b3JzXG4gICAgaWYgKGdyYXBoLmZ1bmN0aW9ucyAhPSBudWxsKSB7XG4gICAgICBPYmplY3Qua2V5cyhncmFwaC5mdW5jdGlvbnMpLmZvckVhY2gobmFtZSA9PiB7XG4gICAgICAgIHRoaXMuX2Z1bmN0aW9uRXhlY3V0b3JNYXBbbmFtZV0gPVxuICAgICAgICAgICAgbmV3IEdyYXBoRXhlY3V0b3IoZ3JhcGguZnVuY3Rpb25zW25hbWVdLCB0aGlzKTtcbiAgICAgIH0pO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgZ2V0Q29tcGlsYXRpb25LZXkoaW5wdXRzOiBOb2RlW10sIG91dHB1dHM6IE5vZGVbXSk6IHN0cmluZyB7XG4gICAgY29uc3Qgc29ydGVkSW5wdXRzID0gaW5wdXRzLm1hcChub2RlID0+IG5vZGUubmFtZSkuc29ydCgpO1xuICAgIGNvbnN0IHNvcnRlZE91dHB1dHMgPSBvdXRwdXRzLm1hcChub2RlID0+IG5vZGUubmFtZSkuc29ydCgpO1xuICAgIHJldHVybiBzb3J0ZWRJbnB1dHMuam9pbih0aGlzLlNFUEVSQVRPUikgKyAnLS0nICtcbiAgICAgICAgc29ydGVkT3V0cHV0cy5qb2luKHRoaXMuU0VQRVJBVE9SKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21waWxlcyB0aGUgaW5mZXJlbmNlIGdyYXBoIGFuZCByZXR1cm5zIHRoZSBtaW5pbWFsIHNldCBvZiBub2RlcyB0aGF0IGFyZVxuICAgKiByZXF1aXJlZCBmb3IgZXhlY3V0aW9uLCBpbiB0aGUgY29ycmVjdCBleGVjdXRpb24gb3JkZXIuXG4gICAqL1xuICBwcml2YXRlIGNvbXBpbGUoaW5wdXRzOiBOYW1lZFRlbnNvck1hcCwgb3V0cHV0czogTm9kZVtdKTogTm9kZVtdIHtcbiAgICBjb25zdCBleGVjdXRpb25JbmZvID1cbiAgICAgICAgZ2V0RXhlY3V0aW9uU3ViZ3JhcGgoaW5wdXRzLCBvdXRwdXRzLCB0aGlzLndlaWdodE1hcCwgdGhpcy5faW5pdE5vZGVzKTtcbiAgICBjb25zdCB7bWlzc2luZ0lucHV0cywgZHluYW1pY05vZGUsIHN5bmNJbnB1dHN9ID0gZXhlY3V0aW9uSW5mbztcbiAgICBpZiAoZHluYW1pY05vZGUgIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBUaGlzIGV4ZWN1dGlvbiBjb250YWlucyB0aGUgbm9kZSAnJHtkeW5hbWljTm9kZS5uYW1lfScsIHdoaWNoIGhhcyBgICtcbiAgICAgICAgICBgdGhlIGR5bmFtaWMgb3AgJyR7ZHluYW1pY05vZGUub3B9Jy4gUGxlYXNlIHVzZSBgICtcbiAgICAgICAgICBgbW9kZWwuZXhlY3V0ZUFzeW5jKCkgaW5zdGVhZC4gQWx0ZXJuYXRpdmVseSwgdG8gYXZvaWQgdGhlIGAgK1xuICAgICAgICAgIGBkeW5hbWljIG9wcywgc3BlY2lmeSB0aGUgaW5wdXRzIFske3N5bmNJbnB1dHN9XWApO1xuICAgIH1cblxuICAgIGlmIChtaXNzaW5nSW5wdXRzLmxlbmd0aCA+IDApIHtcbiAgICAgIGNvbnN0IG91dE5hbWVzID0gb3V0cHV0cy5tYXAobiA9PiBuLm5hbWUpO1xuICAgICAgY29uc3QgaW5OYW1lcyA9IE9iamVjdC5rZXlzKGlucHV0cyk7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYENhbm5vdCBjb21wdXRlIHRoZSBvdXRwdXRzIFske291dE5hbWVzfV0gZnJvbSB0aGUgcHJvdmlkZWQgaW5wdXRzIGAgK1xuICAgICAgICAgIGBbJHtpbk5hbWVzfV0uIE1pc3NpbmcgdGhlIGZvbGxvd2luZyBpbnB1dHM6IFske21pc3NpbmdJbnB1dHN9XWApO1xuICAgIH1cblxuICAgIHJldHVybiBnZXROb2Rlc0luVG9wb2xvZ2ljYWxPcmRlcihcbiAgICAgICAgdGhpcy5ncmFwaCwgdGhpcy53ZWlnaHRNYXAsIGV4ZWN1dGlvbkluZm8pO1xuICB9XG5cbiAgcHJpdmF0ZSBjbG9uZUFuZEtlZXBUZW5zb3IodGVuc29yOiBUZW5zb3IpIHtcbiAgICBpZiAodGVuc29yID09IG51bGwpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICBjb25zdCBjbG9uZSA9IHRlbnNvci5jbG9uZSgpO1xuICAgIC8vIEtlZXAgdGhlIGNsb25lIGJlY2F1c2VgbW9kZWwuZXhlY3V0ZSgpYCBtYXkgYmUgY2FsbGVkIHdpdGhpblxuICAgIC8vIGEgYHRpZHkoKWAsIGJ1dCB0aGUgdXNlciBtYXkgaW5zcGVjdCB0aGVzZSB0ZW5zb3JzIGFmdGVyIHRoZVxuICAgIC8vIHRpZHkuXG4gICAga2VlcChjbG9uZSk7XG4gICAgcmV0dXJuIGNsb25lO1xuICB9XG5cbiAgcHJpdmF0ZSBjbG9uZVRlbnNvckxpc3QodGVuc29yczogVGVuc29yW10pIHtcbiAgICBpZiAoIXRlbnNvcnMpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICBjb25zdCBjbG9uZWRUZW5zb3IgPSB0ZW5zb3JzLm1hcCh0ZW5zb3IgPT4ge1xuICAgICAgcmV0dXJuIHRoaXMuY2xvbmVBbmRLZWVwVGVuc29yKHRlbnNvcik7XG4gICAgfSk7XG4gICAgcmV0dXJuIGNsb25lZFRlbnNvcjtcbiAgfVxuXG4gIHByaXZhdGUgY2xvbmVUZW5zb3JNYXAodGVuc29yc01hcDogTmFtZWRUZW5zb3JzTWFwKTogTmFtZWRUZW5zb3JzTWFwIHtcbiAgICByZXR1cm4gT2JqZWN0LmZyb21FbnRyaWVzKFxuICAgICAgICBPYmplY3QuZW50cmllcyh0ZW5zb3JzTWFwKS5tYXAoKFtuYW1lLCB0ZW5zb3JzTGlzdF0pID0+IHtcbiAgICAgICAgICByZXR1cm4gW25hbWUsIHRoaXMuY2xvbmVUZW5zb3JMaXN0KHRlbnNvcnNMaXN0KV07XG4gICAgICAgIH0pKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBFeGVjdXRlcyB0aGUgaW5mZXJlbmNlIGZvciBnaXZlbiBpbnB1dCB0ZW5zb3JzLlxuICAgKiBAcGFyYW0gaW5wdXRzIFRlbnNvciBtYXAgZm9yIHRoZSBtb2RlbCBpbnB1dHMsIGtleWVkIGJ5IHRoZSBpbnB1dCBub2RlXG4gICAqIG5hbWVzLlxuICAgKiBAcGFyYW0gb3V0cHV0cyBPcHRpb25hbC4gb3V0cHV0IG5vZGUgbmFtZSBmcm9tIHRoZSBUZW5zb3JmbG93IG1vZGVsLCBpZlxuICAgKiBubyBvdXRwdXRzIGFyZSBzcGVjaWZpZWQsIHRoZSBkZWZhdWx0IG91dHB1dHMgb2YgdGhlIG1vZGVsIHdvdWxkIGJlIHVzZWQuXG4gICAqIFlvdSBjYW4gaW5zcGVjdCBpbnRlcm1lZGlhdGUgbm9kZXMgb2YgdGhlIG1vZGVsIGJ5IGFkZGluZyB0aGVtIHRvIHRoZVxuICAgKiBvdXRwdXRzIGFycmF5LlxuICAgKi9cbiAgZXhlY3V0ZShpbnB1dHM6IE5hbWVkVGVuc29yTWFwLCBvdXRwdXRzPzogc3RyaW5nW10pOiBUZW5zb3JbXSB7XG4gICAgLy8gRGlzcG9zZSBhbnkgdGVuc29ycyBmcm9tIGEgcHJpb3IgcnVuIHRvIGF2b2lkIGxlYWtpbmcgdGhlbS5cbiAgICB0aGlzLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JzKCk7XG4gICAgaW5wdXRzID0gdGhpcy5tYXBJbnB1dHMoaW5wdXRzKTtcbiAgICBjb25zdCBuYW1lcyA9IE9iamVjdC5rZXlzKGlucHV0cykuc29ydCgpO1xuICAgIHRoaXMuY2hlY2tJbnB1dHMoaW5wdXRzKTtcbiAgICB0aGlzLmNoZWNrSW5wdXRTaGFwZUFuZFR5cGUoaW5wdXRzKTtcbiAgICBvdXRwdXRzID0gdGhpcy5tYXBPdXRwdXRzKG91dHB1dHMpO1xuICAgIHRoaXMuY2hlY2tPdXRwdXRzKG91dHB1dHMpO1xuICAgIGNvbnN0IGlucHV0Tm9kZXMgPVxuICAgICAgICBuYW1lcy5tYXAobmFtZSA9PiB0aGlzLmdyYXBoLm5vZGVzW3BhcnNlTm9kZU5hbWUobmFtZSlbMF1dKTtcbiAgICBjb25zdCBvdXRwdXROb2RlTmFtZXMgPSBvdXRwdXRzLm1hcChuYW1lID0+IHBhcnNlTm9kZU5hbWUobmFtZSlbMF0pO1xuICAgIGxldCBvdXRwdXROb2RlcyA9IG91dHB1dE5vZGVOYW1lcy5tYXAobmFtZSA9PiB0aGlzLmdyYXBoLm5vZGVzW25hbWVdKTtcbiAgICAvLyBJZiBubyBvdXRwdXRzIGFyZSBzcGVjaWZpZWQsIHRoZW4gdXNlIHRoZSBkZWZhdWx0IG91dHB1dHMgb2YgdGhlIG1vZGVsLlxuICAgIGlmIChvdXRwdXROb2Rlcy5sZW5ndGggPT09IDApIHtcbiAgICAgIG91dHB1dE5vZGVzID0gdGhpcy5fb3V0cHV0cztcbiAgICB9XG5cbiAgICBjb25zdCBjb21waWxhdGlvbktleSA9IHRoaXMuZ2V0Q29tcGlsYXRpb25LZXkoaW5wdXROb2Rlcywgb3V0cHV0Tm9kZXMpO1xuXG4gICAgLy8gRG8gbm90aGluZyBpZiB0aGUgY29tcGlsZWQgZ3JhcGggY2FjaGUgY29udGFpbnMgdGhlIGlucHV0LlxuICAgIGxldCBvcmRlcmVkTm9kZXMgPSB0aGlzLmNvbXBpbGVkTWFwLmdldChjb21waWxhdGlvbktleSk7XG4gICAgaWYgKG9yZGVyZWROb2RlcyA9PSBudWxsKSB7XG4gICAgICBvcmRlcmVkTm9kZXMgPSB0aGlzLmNvbXBpbGUoaW5wdXRzLCBvdXRwdXROb2Rlcyk7XG4gICAgICB0aGlzLmNvbXBpbGVkTWFwLnNldChjb21waWxhdGlvbktleSwgb3JkZXJlZE5vZGVzKTtcbiAgICB9XG5cbiAgICAvLyBLZWVwIHRlbnNvcnMgaWYgS0VFUF9JTlRFUk1FRElBVEVfVEVOU09SUyBpcyBvbi5cbiAgICB0cnkge1xuICAgICAgdGhpcy5rZWVwSW50ZXJtZWRpYXRlVGVuc29ycyA9IGVudigpLmdldEJvb2woJ0tFRVBfSU5URVJNRURJQVRFX1RFTlNPUlMnKTtcbiAgICB9IGNhdGNoIChlKSB7XG4gICAgICB0aGlzLmtlZXBJbnRlcm1lZGlhdGVUZW5zb3JzID0gZmFsc2U7XG4gICAgICBjb25zb2xlLndhcm4oZS5tZXNzYWdlKTtcbiAgICB9XG4gICAgY29uc3QgdGVuc29yQXJyYXlNYXA6IFRlbnNvckFycmF5TWFwID0ge307XG4gICAgY29uc3QgdGVuc29yTGlzdE1hcDogVGVuc29yTGlzdE1hcCA9IHt9O1xuXG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgY29udGV4dCA9IG5ldyBFeGVjdXRpb25Db250ZXh0KFxuICAgICAgICAgIHRoaXMud2VpZ2h0TWFwLCB0ZW5zb3JBcnJheU1hcCwgdGVuc29yTGlzdE1hcCxcbiAgICAgICAgICB0aGlzLmZ1bmN0aW9uRXhlY3V0b3JNYXApO1xuICAgICAgY29uc3QgdGVuc29yc01hcDogTmFtZWRUZW5zb3JzTWFwID0gey4uLnRoaXMud2VpZ2h0TWFwfTtcbiAgICAgIGlmICh0aGlzLmtlZXBJbnRlcm1lZGlhdGVUZW5zb3JzKSB7XG4gICAgICAgIHRoaXMuY2xvbmVkVGVuc29yc01hcCA9IHRoaXMuY2xvbmVUZW5zb3JNYXAodGhpcy53ZWlnaHRNYXApO1xuICAgICAgfVxuXG4gICAgICBPYmplY3Qua2V5cyhpbnB1dHMpLmZvckVhY2gobmFtZSA9PiB7XG4gICAgICAgIGNvbnN0IFtub2RlTmFtZSwgaW5kZXhdID0gcGFyc2VOb2RlTmFtZShuYW1lKTtcbiAgICAgICAgY29uc3QgdGVuc29yczogVGVuc29yW10gPSBbXTtcbiAgICAgICAgdGVuc29yc1tpbmRleF0gPSBpbnB1dHNbbmFtZV07XG4gICAgICAgIHRlbnNvcnNNYXBbbm9kZU5hbWVdID0gdGVuc29ycztcbiAgICAgICAgaWYgKHRoaXMua2VlcEludGVybWVkaWF0ZVRlbnNvcnMpIHtcbiAgICAgICAgICB0aGlzLmNsb25lZFRlbnNvcnNNYXBbbm9kZU5hbWVdID0gdGhpcy5jbG9uZVRlbnNvckxpc3QodGVuc29ycyk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuXG4gICAgICBjb25zdCB0ZW5zb3JzVG9LZWVwID0gdGhpcy5nZXRGcm96ZW5UZW5zb3JJZHModGVuc29yc01hcCk7XG4gICAgICBjb25zdCBpbnRlcm1lZGlhdGVUZW5zb3JDb25zdW1lckNvdW50OiB7W2tleTogbnVtYmVyXTogbnVtYmVyfSA9IHt9O1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBvcmRlcmVkTm9kZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgY29uc3Qgbm9kZSA9IG9yZGVyZWROb2Rlc1tpXTtcbiAgICAgICAgaWYgKCF0ZW5zb3JzTWFwW25vZGUubmFtZV0pIHtcbiAgICAgICAgICBjb25zdCB0ZW5zb3JzID1cbiAgICAgICAgICAgICAgZXhlY3V0ZU9wKG5vZGUsIHRlbnNvcnNNYXAsIGNvbnRleHQsIHRoaXMuX3Jlc291cmNlTWFuYWdlcikgYXNcbiAgICAgICAgICAgICAgVGVuc29yW107XG4gICAgICAgICAgaWYgKHV0aWwuaXNQcm9taXNlKHRlbnNvcnMpKSB7XG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAgICAgYFRoZSBleGVjdXRpb24gb2YgdGhlIG9wICcke25vZGUub3B9JyByZXR1cm5lZCBhIHByb21pc2UuIGAgK1xuICAgICAgICAgICAgICAgIGBQbGVhc2UgdXNlIG1vZGVsLmV4ZWN1dGVBc3luYygpIGluc3RlYWQuYCk7XG4gICAgICAgICAgfVxuICAgICAgICAgIHRlbnNvcnNNYXBbbm9kZS5uYW1lXSA9IHRlbnNvcnM7XG4gICAgICAgICAgaWYgKHRoaXMua2VlcEludGVybWVkaWF0ZVRlbnNvcnMpIHtcbiAgICAgICAgICAgIHRoaXMuY2xvbmVkVGVuc29yc01hcFtub2RlLm5hbWVdID0gdGhpcy5jbG9uZVRlbnNvckxpc3QodGVuc29ycyk7XG4gICAgICAgICAgfVxuICAgICAgICAgIHRoaXMuY2hlY2tUZW5zb3JGb3JEaXNwb3NhbChcbiAgICAgICAgICAgICAgbm9kZS5uYW1lLCBub2RlLCB0ZW5zb3JzTWFwLCBjb250ZXh0LCB0ZW5zb3JzVG9LZWVwLFxuICAgICAgICAgICAgICBvdXRwdXROb2RlTmFtZXMsIGludGVybWVkaWF0ZVRlbnNvckNvbnN1bWVyQ291bnQpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIC8vIGRpc3Bvc2UgdGhlIGNvbnRleHQgZm9yIHRoZSByb290IGV4ZWN1dG9yXG4gICAgICBpZiAodGhpcy5wYXJlbnQgPT0gbnVsbCkge1xuICAgICAgICBjb250ZXh0LmRpc3Bvc2UodGVuc29yc1RvS2VlcCk7XG4gICAgICB9XG5cbiAgICAgIHJldHVybiBvdXRwdXRzLm1hcChuYW1lID0+IGdldFRlbnNvcihuYW1lLCB0ZW5zb3JzTWFwLCBjb250ZXh0KSk7XG4gICAgfSk7XG4gIH1cblxuICBwcml2YXRlIGdldEZyb3plblRlbnNvcklkcyh0ZW5zb3JNYXA6IE5hbWVkVGVuc29yc01hcCk6IFNldDxudW1iZXI+IHtcbiAgICBjb25zdCBpZHMgPSBbXS5jb25jYXQuYXBwbHkoXG4gICAgICAgIFtdLFxuICAgICAgICBPYmplY3Qua2V5cyh0ZW5zb3JNYXApXG4gICAgICAgICAgICAubWFwKGtleSA9PiB0ZW5zb3JNYXBba2V5XSlcbiAgICAgICAgICAgIC5tYXAodGVuc29ycyA9PiB0ZW5zb3JzLm1hcCh0ZW5zb3IgPT4gdGVuc29yLmlkKSkpO1xuICAgIHJldHVybiBuZXcgU2V0KGlkcyk7XG4gIH1cblxuICBwcml2YXRlIGNoZWNrVGVuc29yRm9yRGlzcG9zYWwoXG4gICAgICBub2RlTmFtZTogc3RyaW5nLCBub2RlOiBOb2RlLCB0ZW5zb3JNYXA6IE5hbWVkVGVuc29yc01hcCxcbiAgICAgIGNvbnRleHQ6IEV4ZWN1dGlvbkNvbnRleHQsIHRlbnNvcnNUb0tlZXA6IFNldDxudW1iZXI+LFxuICAgICAgb3V0cHV0TmFtZXM6IHN0cmluZ1tdLFxuICAgICAgaW50ZXJtZWRpYXRlVGVuc29yQ29uc3VtZXJDb3VudDoge1trZXk6IHN0cmluZ106IG51bWJlcn0pIHtcbiAgICAvLyBTa2lwIG91dHB1dCBub2RlcyBhbmQgYW55IGNvbnRyb2wgZmxvdyBub2Rlcywgc2luY2UgaXRzIGRlcGVuZGVuY3kgaXNcbiAgICAvLyB0cmlja3kgdG8gdHJhY2sgY29ycmVjdGx5LlxuICAgIGlmIChub2RlLmNhdGVnb3J5ID09PSAnY29udHJvbCcgfHwgb3V0cHV0TmFtZXMuaW5kZXhPZihub2RlTmFtZSkgIT09IC0xKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdGVuc29yTWFwW25vZGVOYW1lXS5mb3JFYWNoKHRlbnNvciA9PiB7XG4gICAgICBpZiAodGVuc29yICE9IG51bGwpIHtcbiAgICAgICAgaW50ZXJtZWRpYXRlVGVuc29yQ29uc3VtZXJDb3VudFt0ZW5zb3IuaWRdID1cbiAgICAgICAgICAgIChpbnRlcm1lZGlhdGVUZW5zb3JDb25zdW1lckNvdW50W3RlbnNvci5pZF0gfHwgMCkgK1xuICAgICAgICAgICAgbm9kZS5jaGlsZHJlbi5sZW5ndGg7XG4gICAgICB9XG4gICAgfSk7XG4gICAgbm9kZS5pbnB1dHMuZm9yRWFjaChpbnB1dCA9PiB7XG4gICAgICAvLyBTa2lwIGFueSBjb250cm9sIGZsb3cgbm9kZXMsIHNpbmNlIGl0cyBkZXBlbmRlbmN5IGlzIHRyaWNreSB0byB0cmFja1xuICAgICAgLy8gY29ycmVjdGx5LlxuICAgICAgaWYgKGlucHV0LmNhdGVnb3J5ICE9PSAnY29udHJvbCcpIHtcbiAgICAgICAgY29uc3QgdGVuc29ycyA9XG4gICAgICAgICAgICBnZXRUZW5zb3JzRm9yQ3VycmVudENvbnRlbnh0KGlucHV0Lm5hbWUsIHRlbnNvck1hcCwgY29udGV4dCk7XG4gICAgICAgIGlmICh0ZW5zb3JzICE9IG51bGwpIHtcbiAgICAgICAgICB0ZW5zb3JzLmZvckVhY2godGVuc29yID0+IHtcbiAgICAgICAgICAgIGlmICh0ZW5zb3IgJiYgIXRlbnNvci5rZXB0ICYmICF0ZW5zb3JzVG9LZWVwLmhhcyh0ZW5zb3IuaWQpKSB7XG4gICAgICAgICAgICAgIGNvbnN0IGNvdW50ID0gaW50ZXJtZWRpYXRlVGVuc29yQ29uc3VtZXJDb3VudFt0ZW5zb3IuaWRdO1xuICAgICAgICAgICAgICBpZiAoY291bnQgPT09IDEpIHtcbiAgICAgICAgICAgICAgICB0ZW5zb3IuZGlzcG9zZSgpO1xuICAgICAgICAgICAgICAgIGRlbGV0ZSBpbnRlcm1lZGlhdGVUZW5zb3JDb25zdW1lckNvdW50W3RlbnNvci5pZF07XG4gICAgICAgICAgICAgIH0gZWxzZSBpZiAoY291bnQgIT0gbnVsbCkge1xuICAgICAgICAgICAgICAgIC8vIG9ubHkgaW50ZXJtZWRpYXRlIG5vZGVzIGhhcyBjb3VudCBzZXQsIGlucHV0cyBhbmQgd2VpZ2h0c1xuICAgICAgICAgICAgICAgIC8vIGFyZSBub3QuXG4gICAgICAgICAgICAgICAgaW50ZXJtZWRpYXRlVGVuc29yQ29uc3VtZXJDb3VudFt0ZW5zb3IuaWRdLS07XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9KTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIEV4ZWN1dGVzIHRoZSBpbmZlcmVuY2UgZm9yIGdpdmVuIGlucHV0IHRlbnNvcnMgaW4gQXN5bmMgZmFzaGlvbi5cbiAgICogQHBhcmFtIGlucHV0cyBUZW5zb3IgbWFwIGZvciB0aGUgbW9kZWwgaW5wdXRzLCBrZXllZCBieSB0aGUgaW5wdXQgbm9kZVxuICAgKiBuYW1lcy5cbiAgICogQHBhcmFtIG91dHB1dHMgb3V0cHV0IG5vZGUgbmFtZSBmcm9tIHRoZSBUZW5zb3JmbG93IG1vZGVsLCBpZiBubyBvdXRwdXRzXG4gICAqIGFyZSBzcGVjaWZpZWQsIHRoZSBkZWZhdWx0IG91dHB1dHMgb2YgdGhlIG1vZGVsIHdvdWxkIGJlIHVzZWQuIFlvdSBjYW5cbiAgICogaW5zcGVjdCBpbnRlcm1lZGlhdGUgbm9kZXMgb2YgdGhlIG1vZGVsIGJ5IGFkZGluZyB0aGVtIHRvIHRoZSBvdXRwdXRzXG4gICAqIGFycmF5LlxuICAgKi9cbiAgYXN5bmMgZXhlY3V0ZUFzeW5jKGlucHV0czogTmFtZWRUZW5zb3JNYXAsIG91dHB1dHM/OiBzdHJpbmdbXSk6XG4gICAgICBQcm9taXNlPFRlbnNvcltdPiB7XG4gICAgcmV0dXJuIHRoaXMuX2V4ZWN1dGVBc3luYyhpbnB1dHMsIG91dHB1dHMpO1xuICB9XG5cbiAgZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvcnMoKSB7XG4gICAgaWYgKCF0aGlzLmNsb25lZFRlbnNvcnNNYXApIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgT2JqZWN0LnZhbHVlcyh0aGlzLmNsb25lZFRlbnNvcnNNYXApLmZvckVhY2godGVuc29yc0xpc3QgPT4ge1xuICAgICAgZm9yIChjb25zdCB0ZW5zb3Igb2YgdGVuc29yc0xpc3QpIHtcbiAgICAgICAgaWYgKHRlbnNvciAmJiAhdGVuc29yLmlzRGlzcG9zZWQpIHtcbiAgICAgICAgICB0ZW5zb3IuZGlzcG9zZSgpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfSk7XG5cbiAgICB0aGlzLmNsb25lZFRlbnNvcnNNYXAgPSBudWxsO1xuICB9XG5cbiAgZ2V0SW50ZXJtZWRpYXRlVGVuc29ycygpOiBOYW1lZFRlbnNvcnNNYXAge1xuICAgIHJldHVybiB0aGlzLmNsb25lZFRlbnNvcnNNYXA7XG4gIH1cblxuICAvKipcbiAgICogRXhlY3V0ZXMgdGhlIGluZmVyZW5jZSBmb3IgZ2l2ZW4gaW5wdXQgdGVuc29ycyBpbiBBc3luYyBmYXNoaW9uLlxuICAgKiBAcGFyYW0gaW5wdXRzIFRlbnNvciBtYXAgZm9yIHRoZSBtb2RlbCBpbnB1dHMsIGtleWVkIGJ5IHRoZSBpbnB1dCBub2RlXG4gICAqIG5hbWVzLlxuICAgKiBAcGFyYW0gb3V0cHV0cyBPcHRpb25hbC4gb3V0cHV0IG5vZGUgbmFtZSBmcm9tIHRoZSBUZW5zb3JmbG93IG1vZGVsLFxuICAgKiBpZiBubyBvdXRwdXRzIGFyZSBzcGVjaWZpZWQsIHRoZSBkZWZhdWx0IG91dHB1dHMgb2YgdGhlIG1vZGVsIHdvdWxkIGJlXG4gICAqIHVzZWQuIFlvdSBjYW4gaW5zcGVjdCBpbnRlcm1lZGlhdGUgbm9kZXMgb2YgdGhlIG1vZGVsIGJ5IGFkZGluZyB0aGVtIHRvXG4gICAqIHRoZSBvdXRwdXRzIGFycmF5LlxuICAgKiBAcGFyYW0gaXNGdW5jdGlvbkV4ZWN1dGlvbiBPcHRpb25hbC4gRmxhZyBmb3IgZXhlY3V0aW5nIGEgZnVuY3Rpb24uXG4gICAqIEBwYXJhbSB0ZW5zb3JBcnJheU1hcCBPcHRpb25hbCwgZ2xvYmFsIFRlbnNvckFycmF5IG1hcCBieSBpZC4gVXNlZCBmb3JcbiAgICogZnVuY3Rpb24gZXhlY3V0aW9uLlxuICAgKiBAcGFyYW0gdGVuc29yQXJyYXlNYXAgT3B0aW5hbCBnbG9iYWwgVGVuc29yTGlzdCBtYXAgYnkgaWQuIFVzZWQgZm9yXG4gICAqIGZ1bmN0aW9uIGV4ZWN1dGlvbi5cbiAgICovXG4gIHByaXZhdGUgYXN5bmMgX2V4ZWN1dGVBc3luYyhcbiAgICAgIGlucHV0czogTmFtZWRUZW5zb3JNYXAsIG91dHB1dHM/OiBzdHJpbmdbXSwgaXNGdW5jdGlvbkV4ZWN1dGlvbiA9IGZhbHNlLFxuICAgICAgdGVuc29yQXJyYXlNYXA6IFRlbnNvckFycmF5TWFwID0ge30sXG4gICAgICB0ZW5zb3JMaXN0TWFwOiBUZW5zb3JMaXN0TWFwID0ge30pOiBQcm9taXNlPFRlbnNvcltdPiB7XG4gICAgLy8gRGlzcG9zZSBhbnkgdGVuc29ycyBmcm9tIGEgcHJpb3IgcnVuIHRvIGF2b2lkIGxlYWtpbmcgdGhlbS5cbiAgICB0aGlzLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JzKCk7XG4gICAgaWYgKCFpc0Z1bmN0aW9uRXhlY3V0aW9uKSB7XG4gICAgICBpbnB1dHMgPSB0aGlzLm1hcElucHV0cyhpbnB1dHMpO1xuICAgICAgdGhpcy5jaGVja0lucHV0cyhpbnB1dHMpO1xuICAgICAgdGhpcy5jaGVja0lucHV0U2hhcGVBbmRUeXBlKGlucHV0cyk7XG4gICAgICBvdXRwdXRzID0gdGhpcy5tYXBPdXRwdXRzKG91dHB1dHMpO1xuICAgICAgdGhpcy5jaGVja091dHB1dHMob3V0cHV0cyk7XG4gICAgfVxuXG4gICAgLy8gS2VlcCB0ZW5zb3JzIGlmIEtFRVBfSU5URVJNRURJQVRFX1RFTlNPUlMgaXMgb24uXG4gICAgdHJ5IHtcbiAgICAgIHRoaXMua2VlcEludGVybWVkaWF0ZVRlbnNvcnMgPSBlbnYoKS5nZXRCb29sKCdLRUVQX0lOVEVSTUVESUFURV9URU5TT1JTJyk7XG4gICAgfSBjYXRjaCAoZSkge1xuICAgICAgdGhpcy5rZWVwSW50ZXJtZWRpYXRlVGVuc29ycyA9IGZhbHNlO1xuICAgICAgY29uc29sZS53YXJuKGUubWVzc2FnZSk7XG4gICAgfVxuXG4gICAgY29uc3QgY29udGV4dCA9IG5ldyBFeGVjdXRpb25Db250ZXh0KFxuICAgICAgICB0aGlzLndlaWdodE1hcCwgdGVuc29yQXJyYXlNYXAsIHRlbnNvckxpc3RNYXAsXG4gICAgICAgIHRoaXMuZnVuY3Rpb25FeGVjdXRvck1hcCk7XG5cbiAgICBpZiAodGhpcy5rZWVwSW50ZXJtZWRpYXRlVGVuc29ycykge1xuICAgICAgdGhpcy5jbG9uZWRUZW5zb3JzTWFwID0gdGhpcy5jbG9uZVRlbnNvck1hcCh0aGlzLndlaWdodE1hcCk7XG4gICAgfVxuXG4gICAgLy8gR3JhcGggd2l0aCBjb250cm9sIGZsb3cgb3AgcmVxdWlyZXMgcnVudGltZSBldmFsdWF0aW9uIG9mIHRoZSBleGVjdXRpb25cbiAgICAvLyBvcmRlciwgd2hpbGUgd2l0aG91dCBjb250cm9sIGZsb3cgdGhlIGV4ZWN1dGlvbiBvcmRlciBpcyBwcmUtZGV0ZXJtaW5lZFxuICAgIC8vIGluIHRoZSBjb21waWxlIG1ldGhvZC5cbiAgICBjb25zdCB0ZW5zb3JzTWFwID0gYXdhaXQgdGhpcy5leGVjdXRlV2l0aENvbnRyb2xGbG93KFxuICAgICAgICBpbnB1dHMsIGNvbnRleHQsIG91dHB1dHMsIGlzRnVuY3Rpb25FeGVjdXRpb24pO1xuICAgIGNvbnN0IHJlc3VsdHMgPSBvdXRwdXRzLm1hcChuYW1lID0+IGdldFRlbnNvcihuYW1lLCB0ZW5zb3JzTWFwLCBjb250ZXh0KSk7XG5cbiAgICAvLyBkaXNwb3NlIGFsbCB0aGUgaW50ZXJtZWRpYXRlIHRlbnNvcnNcbiAgICBjb25zdCBvdXRwdXRJZHMgPSByZXN1bHRzLm1hcCh0ID0+IHQuaWQpO1xuICAgIGNvbnN0IGlucHV0SWRzID0gT2JqZWN0LmtleXMoaW5wdXRzKS5tYXAobmFtZSA9PiBpbnB1dHNbbmFtZV0uaWQpO1xuICAgIGNvbnN0IGtlZXBJZHMgPVxuICAgICAgICBuZXcgU2V0PG51bWJlcj4oWy4uLm91dHB1dElkcywgLi4uaW5wdXRJZHMsIC4uLnRoaXMud2VpZ2h0SWRzXSk7XG5cbiAgICBPYmplY3QudmFsdWVzKHRlbnNvcnNNYXApLmZvckVhY2godGVuc29yc0xpc3QgPT4ge1xuICAgICAgdGVuc29yc0xpc3QuZm9yRWFjaCh0ZW5zb3IgPT4ge1xuICAgICAgICBpZiAodGVuc29yICYmICF0ZW5zb3IuaXNEaXNwb3NlZCAmJiAha2VlcElkcy5oYXModGVuc29yLmlkKSkge1xuICAgICAgICAgIHRlbnNvci5kaXNwb3NlKCk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH0pO1xuXG4gICAgLy8gZGlzcG9zZSB0aGUgY29udGV4dCBmb3IgdGhlIHJvb3QgZXhlY3V0b3JcbiAgICBpZiAodGhpcy5wYXJlbnQgPT0gbnVsbCkge1xuICAgICAgY29udGV4dC5kaXNwb3NlKGtlZXBJZHMpO1xuICAgIH1cblxuICAgIHJldHVybiByZXN1bHRzO1xuICB9XG5cbiAgYXN5bmMgZXhlY3V0ZUZ1bmN0aW9uQXN5bmMoXG4gICAgICBpbnB1dHM6IFRlbnNvcltdLCB0ZW5zb3JBcnJheU1hcDogVGVuc29yQXJyYXlNYXAsXG4gICAgICB0ZW5zb3JMaXN0TWFwOiBUZW5zb3JMaXN0TWFwKTogUHJvbWlzZTxUZW5zb3JbXT4ge1xuICAgIGNvbnN0IG1hcHBlZElucHV0cyA9IGlucHV0cy5yZWR1Y2UoKG1hcCwgdGVuc29yLCBpbmRleCkgPT4ge1xuICAgICAgbWFwW3RoaXMuaW5wdXRzW2luZGV4XS5uYW1lXSA9IHRlbnNvcjtcbiAgICAgIHJldHVybiBtYXA7XG4gICAgfSwge30gYXMgTmFtZWRUZW5zb3JNYXApO1xuXG4gICAgcmV0dXJuIHRoaXMuX2V4ZWN1dGVBc3luYyhcbiAgICAgICAgbWFwcGVkSW5wdXRzLCB0aGlzLm91dHB1dE5vZGVzLCB0cnVlLCB0ZW5zb3JBcnJheU1hcCwgdGVuc29yTGlzdE1hcCk7XG4gIH1cblxuICAvKipcbiAgICogV2hlbiB0aGVyZSBhcmUgY29udHJvbCBmbG93IG5vZGVzIGluIHRoZSBncmFwaCwgdGhlIGdyYXBoIGV4ZWN1dGlvbiB1c2VcbiAgICogRXhlY3V0aW9uQ29udGV4dCB0byBrZWVwIHRyYWNrIG9mIHRoZSBmcmFtZXMgYW5kIGxvb3AgaXRlcmF0b3JzLlxuICAgKiBAcGFyYW0gaW5wdXRzIHBsYWNlaG9sZGVyIHRlbnNvcnMgZm9yIHRoZSBncmFwaC5cbiAgICogQHBhcmFtIGNvbnRleHQgdGhlIGV4ZWN1dGlvbiBjb250ZXh0IG9iamVjdCBmb3IgY3VycmVudCBleGVjdXRpb24uXG4gICAqIEBwYXJhbSBvdXRwdXROYW1lcyBPcHRpb25hbC4gb3V0cHV0IG5vZGUgbmFtZSBmcm9tIHRoZSBUZW5zb3JmbG93IG1vZGVsLFxuICAgKiBpZiBubyBvdXRwdXRzIGFyZSBzcGVjaWZpZWQsIHRoZSBkZWZhdWx0IG91dHB1dHMgb2YgdGhlIG1vZGVsIHdvdWxkIGJlXG4gICAqIHVzZWQuIFlvdSBjYW4gaW5zcGVjdCBpbnRlcm1lZGlhdGUgbm9kZXMgb2YgdGhlIG1vZGVsIGJ5IGFkZGluZyB0aGVtIHRvXG4gICAqIHRoZSBvdXRwdXRzIGFycmF5LlxuICAgKiBAcGFyYW0gaXNGdW5jdGlvbkV4ZWN1dGlvbiBGbGFnIGZvciBleGVjdXRpbmcgYSBmdW5jdGlvbi5cbiAgICovXG4gIHByaXZhdGUgYXN5bmMgZXhlY3V0ZVdpdGhDb250cm9sRmxvdyhcbiAgICAgIGlucHV0czogTmFtZWRUZW5zb3JNYXAsIGNvbnRleHQ6IEV4ZWN1dGlvbkNvbnRleHQsIG91dHB1dE5hbWVzPzogc3RyaW5nW10sXG4gICAgICBpc0Z1bmN0aW9uRXhlY3V0aW9uPzogYm9vbGVhbik6IFByb21pc2U8TmFtZWRUZW5zb3JzTWFwPiB7XG4gICAgY29uc3QgbmFtZXMgPSBPYmplY3Qua2V5cyhpbnB1dHMpO1xuICAgIGNvbnN0IGlucHV0Tm9kZXMgPVxuICAgICAgICBuYW1lcy5tYXAobmFtZSA9PiB0aGlzLmdyYXBoLm5vZGVzW3BhcnNlTm9kZU5hbWUobmFtZSlbMF1dKTtcbiAgICBjb25zdCBvdXRwdXROb2RlTmFtZXMgPSBvdXRwdXROYW1lcy5tYXAobmFtZSA9PiBwYXJzZU5vZGVOYW1lKG5hbWUpWzBdKTtcbiAgICBsZXQgb3V0cHV0Tm9kZXMgPSBvdXRwdXROb2RlTmFtZXMubWFwKG5hbWUgPT4gdGhpcy5ncmFwaC5ub2Rlc1tuYW1lXSk7XG5cbiAgICAvLyBJZiBubyBvdXRwdXRzIGFyZSBzcGVjaWZpZWQsIHRoZW4gdXNlIHRoZSBkZWZhdWx0IG91dHB1dHMgb2YgdGhlIG1vZGVsLlxuICAgIGlmIChvdXRwdXROb2Rlcy5sZW5ndGggPT09IDApIHtcbiAgICAgIG91dHB1dE5vZGVzID0gdGhpcy5fb3V0cHV0cztcbiAgICB9XG5cbiAgICBjb25zdCB7dXNlZE5vZGVzLCBtaXNzaW5nSW5wdXRzLCBkeW5hbWljTm9kZSwgc3luY0lucHV0c30gPVxuICAgICAgICBnZXRFeGVjdXRpb25TdWJncmFwaChcbiAgICAgICAgICAgIGlucHV0cywgb3V0cHV0Tm9kZXMsIHRoaXMud2VpZ2h0TWFwLCB0aGlzLl9pbml0Tm9kZXMpO1xuXG4gICAgLy8gRmlyc3Qgbm9kZXMgdG8gZXhlY3V0ZSBpbmNsdWRlIGlucHV0Tm9kZXMsIHdlaWdodHMsIGFuZCBpbml0Tm9kZXMuXG4gICAgY29uc3Qgc3RhY2s6IE5vZGVXaXRoQ29udGV4dHNbXSA9IFtcbiAgICAgIC4uLmlucHV0Tm9kZXMsIC4uLnRoaXMuZ3JhcGgud2VpZ2h0cywgLi4uKHRoaXMuX2luaXROb2RlcyB8fCBbXSlcbiAgICBdLm1hcChub2RlID0+IHtcbiAgICAgIHJldHVybiB7bm9kZSwgY29udGV4dHM6IGNvbnRleHQuY3VycmVudENvbnRleHR9O1xuICAgIH0pO1xuICAgIGNvbnN0IHRlbnNvcnNNYXA6IE5hbWVkVGVuc29yc01hcCA9IHsuLi50aGlzLndlaWdodE1hcH07XG4gICAgT2JqZWN0LmtleXMoaW5wdXRzKS5mb3JFYWNoKG5hbWUgPT4ge1xuICAgICAgY29uc3QgW25vZGVOYW1lLCBpbmRleF0gPSBwYXJzZU5vZGVOYW1lKG5hbWUpO1xuICAgICAgY29uc3QgdGVuc29yczogVGVuc29yW10gPSBbXTtcbiAgICAgIHRlbnNvcnNbaW5kZXhdID0gaW5wdXRzW25hbWVdO1xuICAgICAgdGVuc29yc01hcFtub2RlTmFtZV0gPSB0ZW5zb3JzO1xuICAgIH0pO1xuICAgIGNvbnN0IGludGVybWVkaWF0ZVRlbnNvckNvbnN1bWVyQ291bnQ6IHtba2V5OiBudW1iZXJdOiBudW1iZXJ9ID0ge307XG4gICAgY29uc3QgdGVuc29yc1RvS2VlcCA9IHRoaXMuZ2V0RnJvemVuVGVuc29ySWRzKHRlbnNvcnNNYXApO1xuICAgIGNvbnN0IGFkZGVkOiB7W2tleTogc3RyaW5nXTogYm9vbGVhbn0gPSB7fTtcbiAgICB3aGlsZSAoc3RhY2subGVuZ3RoID4gMCkge1xuICAgICAgY29uc3QgcHJvbWlzZXMgPSB0aGlzLnByb2Nlc3NTdGFjayhcbiAgICAgICAgICBpbnB1dE5vZGVzLCBzdGFjaywgY29udGV4dCwgdGVuc29yc01hcCwgYWRkZWQsIHRlbnNvcnNUb0tlZXAsXG4gICAgICAgICAgb3V0cHV0Tm9kZU5hbWVzLCBpbnRlcm1lZGlhdGVUZW5zb3JDb25zdW1lckNvdW50LCB1c2VkTm9kZXMpO1xuICAgICAgYXdhaXQgUHJvbWlzZS5hbGwocHJvbWlzZXMpO1xuICAgIH1cbiAgICBpZiAoZHluYW1pY05vZGUgPT0gbnVsbCAmJiAhaXNGdW5jdGlvbkV4ZWN1dGlvbikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBUaGlzIG1vZGVsIGV4ZWN1dGlvbiBkaWQgbm90IGNvbnRhaW4gYW55IG5vZGVzIHdpdGggY29udHJvbCBmbG93IGAgK1xuICAgICAgICAgIGBvciBkeW5hbWljIG91dHB1dCBzaGFwZXMuIFlvdSBjYW4gdXNlIG1vZGVsLmV4ZWN1dGUoKSBpbnN0ZWFkLmApO1xuICAgIH1cbiAgICBjb25zdCBtaXNzaW5nT3V0cHV0cyA9XG4gICAgICAgIG91dHB1dE5vZGVzXG4gICAgICAgICAgICAuZmlsdGVyKFxuICAgICAgICAgICAgICAgIG5vZGUgPT4gIWlzQ29udHJvbEZsb3cobm9kZSkgJiZcbiAgICAgICAgICAgICAgICAgICAgIWdldFRlbnNvcihub2RlLm5hbWUsIHRlbnNvcnNNYXAsIGNvbnRleHQpKVxuICAgICAgICAgICAgLm1hcChub2RlID0+IG5vZGUubmFtZSk7XG4gICAgaWYgKG1pc3NpbmdPdXRwdXRzLmxlbmd0aCA+IDApIHtcbiAgICAgIGxldCBhbHRlcm5hdGl2ZU1zZyA9ICcnO1xuICAgICAgaWYgKGR5bmFtaWNOb2RlICE9IG51bGwpIHtcbiAgICAgICAgYWx0ZXJuYXRpdmVNc2cgPVxuICAgICAgICAgICAgYEFsdGVybmF0aXZlbHksIHRvIGF2b2lkIHRoZSBkeW5hbWljIG9wcywgdXNlIG1vZGVsLmV4ZWN1dGUoKSBgICtcbiAgICAgICAgICAgIGBhbmQgc3BlY2lmeSB0aGUgaW5wdXRzIFske3N5bmNJbnB1dHN9XWA7XG4gICAgICB9XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYENhbm5vdCBjb21wdXRlIHRoZSBvdXRwdXRzIFske21pc3NpbmdPdXRwdXRzfV0gZnJvbSB0aGUgcHJvdmlkZWQgYCArXG4gICAgICAgICAgYGlucHV0cyBbJHtuYW1lc31dLiBDb25zaWRlciBwcm92aWRpbmcgdGhlIGZvbGxvd2luZyBpbnB1dHM6IGAgK1xuICAgICAgICAgIGBbJHttaXNzaW5nSW5wdXRzfV0uICR7YWx0ZXJuYXRpdmVNc2d9YCk7XG4gICAgfVxuICAgIHJldHVybiB0ZW5zb3JzTWFwO1xuICB9XG5cbiAgcHJpdmF0ZSBwcm9jZXNzU3RhY2soXG4gICAgICBpbnB1dE5vZGVzOiBOb2RlW10sIHN0YWNrOiBOb2RlV2l0aENvbnRleHRzW10sIGNvbnRleHQ6IEV4ZWN1dGlvbkNvbnRleHQsXG4gICAgICB0ZW5zb3JNYXA6IE5hbWVkVGVuc29yc01hcCwgYWRkZWQ6IHtba2V5OiBzdHJpbmddOiBib29sZWFufSxcbiAgICAgIHRlbnNvcnNUb0tlZXA6IFNldDxudW1iZXI+LCBvdXRwdXROYW1lczogc3RyaW5nW10sXG4gICAgICBpbnRlcm1lZGlhdGVUZW5zb3JDb25zdW1lckNvdW50OiB7W2tleTogbnVtYmVyXTogbnVtYmVyfSxcbiAgICAgIHVzZWROb2RlczogU2V0PHN0cmluZz4pIHtcbiAgICBjb25zdCBwcm9taXNlczogQXJyYXk8UHJvbWlzZTxUZW5zb3JbXT4+ID0gW107XG4gICAgd2hpbGUgKHN0YWNrLmxlbmd0aCA+IDApIHtcbiAgICAgIGNvbnN0IGl0ZW0gPSBzdGFjay5wb3AoKTtcbiAgICAgIGNvbnRleHQuY3VycmVudENvbnRleHQgPSBpdGVtLmNvbnRleHRzO1xuICAgICAgbGV0IG5vZGVOYW1lID0gJyc7XG4gICAgICAvLyBUaGUgdGVuc29yIG9mIHRoZSBFbnRlciBvcCB3aXRoIGlzQ29uc3RhbnQgc2V0IHNob3VsZCBiZSBzZXRcbiAgICAgIC8vIGluIHRoZSBwYXJlbnQgc2NvcGUsIHNvIGl0IHdpbGwgYmUgYXZhaWxhYmxlIGFzIGNvbnN0YW50IGZvciB0aGVcbiAgICAgIC8vIHdob2xlIGxvb3AuXG4gICAgICBpZiAoaXRlbS5ub2RlLm9wID09PSAnRW50ZXInICYmXG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnaXNDb25zdGFudCcsIGl0ZW0ubm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSkge1xuICAgICAgICBbbm9kZU5hbWVdID0gZ2V0Tm9kZU5hbWVBbmRJbmRleChpdGVtLm5vZGUubmFtZSwgY29udGV4dCk7XG4gICAgICB9XG5cbiAgICAgIC8vIG9ubHkgcHJvY2VzcyBub2RlcyB0aGF0IGFyZSBub3QgaW4gdGhlIHRlbnNvck1hcCB5ZXQsIHRoaXMgaW5jbHVkZVxuICAgICAgLy8gaW5wdXROb2RlcyBhbmQgaW50ZXJuYWwgaW5pdE5vZGVzLlxuICAgICAgaWYgKHRlbnNvck1hcFtpdGVtLm5vZGUubmFtZV0gPT0gbnVsbCkge1xuICAgICAgICBjb25zdCB0ZW5zb3JzID1cbiAgICAgICAgICAgIGV4ZWN1dGVPcChpdGVtLm5vZGUsIHRlbnNvck1hcCwgY29udGV4dCwgdGhpcy5fcmVzb3VyY2VNYW5hZ2VyKTtcbiAgICAgICAgaWYgKCFub2RlTmFtZSkge1xuICAgICAgICAgIFtub2RlTmFtZV0gPSBnZXROb2RlTmFtZUFuZEluZGV4KGl0ZW0ubm9kZS5uYW1lLCBjb250ZXh0KTtcbiAgICAgICAgfVxuICAgICAgICBjb25zdCBjdXJyZW50Q29udGV4dCA9IGNvbnRleHQuY3VycmVudENvbnRleHQ7XG4gICAgICAgIGlmICh1dGlsLmlzUHJvbWlzZSh0ZW5zb3JzKSkge1xuICAgICAgICAgIHByb21pc2VzLnB1c2godGVuc29ycy50aGVuKHQgPT4ge1xuICAgICAgICAgICAgdGVuc29yTWFwW25vZGVOYW1lXSA9IHQ7XG4gICAgICAgICAgICBpZiAodGhpcy5rZWVwSW50ZXJtZWRpYXRlVGVuc29ycykge1xuICAgICAgICAgICAgICB0aGlzLmNsb25lZFRlbnNvcnNNYXBbbm9kZU5hbWVdID0gdGhpcy5jbG9uZVRlbnNvckxpc3QodCk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBjb250ZXh0LmN1cnJlbnRDb250ZXh0ID0gY3VycmVudENvbnRleHQ7XG4gICAgICAgICAgICB0aGlzLmNoZWNrVGVuc29yRm9yRGlzcG9zYWwoXG4gICAgICAgICAgICAgICAgbm9kZU5hbWUsIGl0ZW0ubm9kZSwgdGVuc29yTWFwLCBjb250ZXh0LCB0ZW5zb3JzVG9LZWVwLFxuICAgICAgICAgICAgICAgIG91dHB1dE5hbWVzLCBpbnRlcm1lZGlhdGVUZW5zb3JDb25zdW1lckNvdW50KTtcbiAgICAgICAgICAgIHRoaXMucHJvY2Vzc0NoaWxkTm9kZXMoXG4gICAgICAgICAgICAgICAgaXRlbS5ub2RlLCBzdGFjaywgY29udGV4dCwgdGVuc29yTWFwLCBhZGRlZCwgdXNlZE5vZGVzKTtcbiAgICAgICAgICAgIHJldHVybiB0O1xuICAgICAgICAgIH0pKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0ZW5zb3JNYXBbbm9kZU5hbWVdID0gdGVuc29ycztcbiAgICAgICAgICBpZiAodGhpcy5rZWVwSW50ZXJtZWRpYXRlVGVuc29ycykge1xuICAgICAgICAgICAgdGhpcy5jbG9uZWRUZW5zb3JzTWFwW25vZGVOYW1lXSA9IHRoaXMuY2xvbmVUZW5zb3JMaXN0KHRlbnNvcnMpO1xuICAgICAgICAgIH1cbiAgICAgICAgICB0aGlzLmNoZWNrVGVuc29yRm9yRGlzcG9zYWwoXG4gICAgICAgICAgICAgIG5vZGVOYW1lLCBpdGVtLm5vZGUsIHRlbnNvck1hcCwgY29udGV4dCwgdGVuc29yc1RvS2VlcCxcbiAgICAgICAgICAgICAgb3V0cHV0TmFtZXMsIGludGVybWVkaWF0ZVRlbnNvckNvbnN1bWVyQ291bnQpO1xuICAgICAgICAgIHRoaXMucHJvY2Vzc0NoaWxkTm9kZXMoXG4gICAgICAgICAgICAgIGl0ZW0ubm9kZSwgc3RhY2ssIGNvbnRleHQsIHRlbnNvck1hcCwgYWRkZWQsIHVzZWROb2Rlcyk7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRoaXMucHJvY2Vzc0NoaWxkTm9kZXMoXG4gICAgICAgICAgICBpdGVtLm5vZGUsIHN0YWNrLCBjb250ZXh0LCB0ZW5zb3JNYXAsIGFkZGVkLCB1c2VkTm9kZXMpO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gcHJvbWlzZXM7XG4gIH1cblxuICBwcml2YXRlIHByb2Nlc3NDaGlsZE5vZGVzKFxuICAgICAgbm9kZTogTm9kZSwgc3RhY2s6IE5vZGVXaXRoQ29udGV4dHNbXSwgY29udGV4dDogRXhlY3V0aW9uQ29udGV4dCxcbiAgICAgIHRlbnNvck1hcDogTmFtZWRUZW5zb3JzTWFwLCBhZGRlZDoge1trZXk6IHN0cmluZ106IGJvb2xlYW59LFxuICAgICAgdXNlZE5vZGVzOiBTZXQ8c3RyaW5nPikge1xuICAgIG5vZGUuY2hpbGRyZW4uZm9yRWFjaCgoY2hpbGROb2RlKSA9PiB7XG4gICAgICBjb25zdCBbbm9kZU5hbWUsIF0gPSBnZXROb2RlTmFtZUFuZEluZGV4KGNoaWxkTm9kZS5uYW1lLCBjb250ZXh0KTtcbiAgICAgIGlmIChhZGRlZFtub2RlTmFtZV0gfHwgIXVzZWROb2Rlcy5oYXMoY2hpbGROb2RlLm5hbWUpKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIC8vIE1lcmdlIG9wIGNhbiBiZSBwdXNoZWQgaWYgYW55IG9mIGl0cyBpbnB1dHMgaGFzIHZhbHVlLlxuICAgICAgaWYgKGNoaWxkTm9kZS5vcCA9PT0gJ01lcmdlJykge1xuICAgICAgICBpZiAoY2hpbGROb2RlLmlucHV0TmFtZXMuc29tZShuYW1lID0+IHtcbiAgICAgICAgICAgICAgcmV0dXJuICEhZ2V0VGVuc29yKG5hbWUsIHRlbnNvck1hcCwgY29udGV4dCk7XG4gICAgICAgICAgICB9KSkge1xuICAgICAgICAgIGFkZGVkW25vZGVOYW1lXSA9IHRydWU7XG4gICAgICAgICAgc3RhY2sucHVzaCh7Y29udGV4dHM6IGNvbnRleHQuY3VycmVudENvbnRleHQsIG5vZGU6IGNoaWxkTm9kZX0pO1xuICAgICAgICB9XG4gICAgICB9IGVsc2UgIC8vIE90aGVyd2lzZSBhbGwgaW5wdXRzIG11c3QgdG8gaGF2ZSB2YWx1ZS5cbiAgICAgICAgICBpZiAoY2hpbGROb2RlLmlucHV0TmFtZXMuZXZlcnkobmFtZSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuICEhZ2V0VGVuc29yKG5hbWUsIHRlbnNvck1hcCwgY29udGV4dCk7XG4gICAgICAgICAgICAgIH0pKSB7XG4gICAgICAgIGFkZGVkW25vZGVOYW1lXSA9IHRydWU7XG4gICAgICAgIHN0YWNrLnB1c2goe2NvbnRleHRzOiBjb250ZXh0LmN1cnJlbnRDb250ZXh0LCBub2RlOiBjaGlsZE5vZGV9KTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSZWxlYXNlcyB0aGUgbWVtb3J5IHVzZWQgYnkgdGhlIHdlaWdodCB0ZW5zb3JzLlxuICAgKi9cbiAgZGlzcG9zZSgpIHtcbiAgICBPYmplY3Qua2V5cyh0aGlzLndlaWdodE1hcClcbiAgICAgICAgLmZvckVhY2goXG4gICAgICAgICAgICBrZXkgPT4gdGhpcy53ZWlnaHRNYXBba2V5XS5mb3JFYWNoKHRlbnNvciA9PiB0ZW5zb3IuZGlzcG9zZSgpKSk7XG4gIH1cblxuICBwcml2YXRlIGNoZWNrSW5wdXRTaGFwZUFuZFR5cGUoaW5wdXRzOiBOYW1lZFRlbnNvck1hcCkge1xuICAgIE9iamVjdC5rZXlzKGlucHV0cykuZm9yRWFjaChuYW1lID0+IHtcbiAgICAgIGNvbnN0IGlucHV0ID0gaW5wdXRzW25hbWVdO1xuICAgICAgY29uc3QgW25vZGVOYW1lLCBdID0gcGFyc2VOb2RlTmFtZShuYW1lKTtcbiAgICAgIGNvbnN0IG5vZGUgPSB0aGlzLmdyYXBoLm5vZGVzW25vZGVOYW1lXTtcbiAgICAgIGlmIChub2RlLmF0dHJQYXJhbXNbJ3NoYXBlJ10gJiYgbm9kZS5hdHRyUGFyYW1zWydzaGFwZSddLnZhbHVlKSB7XG4gICAgICAgIGNvbnN0IHNoYXBlID0gbm9kZS5hdHRyUGFyYW1zWydzaGFwZSddLnZhbHVlIGFzIG51bWJlcltdO1xuICAgICAgICBjb25zdCBtYXRjaCA9IHNoYXBlLmxlbmd0aCA9PT0gaW5wdXQuc2hhcGUubGVuZ3RoICYmXG4gICAgICAgICAgICBpbnB1dC5zaGFwZS5ldmVyeShcbiAgICAgICAgICAgICAgICAoZGltLCBpbmRleCkgPT4gc2hhcGVbaW5kZXhdID09PSAtMSB8fCBzaGFwZVtpbmRleF0gPT09IGRpbSk7XG4gICAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgICAgbWF0Y2gsXG4gICAgICAgICAgICAoKSA9PiBgVGhlIHNoYXBlIG9mIGRpY3RbJyR7bm9kZS5uYW1lfSddIHByb3ZpZGVkIGluIGAgK1xuICAgICAgICAgICAgICAgIGBtb2RlbC5leGVjdXRlKGRpY3QpIG11c3QgYmUgWyR7c2hhcGV9XSwgYnV0IHdhcyBgICtcbiAgICAgICAgICAgICAgICBgWyR7aW5wdXQuc2hhcGV9XWApO1xuICAgICAgfVxuICAgICAgaWYgKG5vZGUuYXR0clBhcmFtc1snZHR5cGUnXSAmJiBub2RlLmF0dHJQYXJhbXNbJ2R0eXBlJ10udmFsdWUpIHtcbiAgICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgICBpbnB1dC5kdHlwZSA9PT0gbm9kZS5hdHRyUGFyYW1zWydkdHlwZSddLnZhbHVlIGFzIHN0cmluZyxcbiAgICAgICAgICAgICgpID0+IGBUaGUgZHR5cGUgb2YgZGljdFsnJHtub2RlLm5hbWV9J10gcHJvdmlkZWQgaW4gYCArXG4gICAgICAgICAgICAgICAgYG1vZGVsLmV4ZWN1dGUoZGljdCkgbXVzdCBiZSBgICtcbiAgICAgICAgICAgICAgICBgJHtub2RlLmF0dHJQYXJhbXNbJ2R0eXBlJ10udmFsdWV9LCBidXQgd2FzICR7aW5wdXQuZHR5cGV9YCk7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICBwcml2YXRlIG1hcElucHV0cyhpbnB1dHM6IE5hbWVkVGVuc29yTWFwKSB7XG4gICAgY29uc3QgcmVzdWx0OiBOYW1lZFRlbnNvck1hcCA9IHt9O1xuICAgIGZvciAoY29uc3QgaW5wdXROYW1lIGluIGlucHV0cykge1xuICAgICAgY29uc3QgdGVuc29yID0gdGhpcy5fc2lnbmF0dXJlID8uaW5wdXRzID8uW2lucHV0TmFtZV07XG4gICAgICBpZiAodGVuc29yICE9IG51bGwpIHtcbiAgICAgICAgcmVzdWx0W3RlbnNvci5uYW1lXSA9IGlucHV0c1tpbnB1dE5hbWVdO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmVzdWx0W2lucHV0TmFtZV0gPSBpbnB1dHNbaW5wdXROYW1lXTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByaXZhdGUgY2hlY2tJbnB1dHMoaW5wdXRzOiBOYW1lZFRlbnNvck1hcCkge1xuICAgIGNvbnN0IG5vdEluR3JhcGggPSBPYmplY3Qua2V5cyhpbnB1dHMpLmZpbHRlcihuYW1lID0+IHtcbiAgICAgIGNvbnN0IFtub2RlTmFtZV0gPSBwYXJzZU5vZGVOYW1lKG5hbWUpO1xuICAgICAgcmV0dXJuIHRoaXMuZ3JhcGgubm9kZXNbbm9kZU5hbWVdID09IG51bGw7XG4gICAgfSk7XG4gICAgaWYgKG5vdEluR3JhcGgubGVuZ3RoID4gMCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBUaGUgZGljdCBwcm92aWRlZCBpbiBtb2RlbC5leGVjdXRlKGRpY3QpIGhhcyBgICtcbiAgICAgICAgICBga2V5czogWyR7bm90SW5HcmFwaH1dIHRoYXQgYXJlIG5vdCBwYXJ0IG9mIGdyYXBoYCk7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBtYXBPdXRwdXRzKG91dHB1dHM6IHN0cmluZ1tdKSB7XG4gICAgcmV0dXJuIG91dHB1dHMubWFwKG5hbWUgPT4ge1xuICAgICAgY29uc3QgdGVuc29yID0gdGhpcy5fc2lnbmF0dXJlID8ub3V0cHV0cyA/LltuYW1lXTtcbiAgICAgIGlmICh0ZW5zb3IgIT0gbnVsbCkge1xuICAgICAgICByZXR1cm4gdGVuc29yLm5hbWU7XG4gICAgICB9XG4gICAgICByZXR1cm4gbmFtZTtcbiAgICB9LCB7fSk7XG4gIH1cblxuICBwcml2YXRlIGNoZWNrT3V0cHV0cyhvdXRwdXRzOiBzdHJpbmdbXSk6IHZvaWQge1xuICAgIG91dHB1dHMuZm9yRWFjaChuYW1lID0+IHtcbiAgICAgIGNvbnN0IFtub3JtYWxpemVkTmFtZV0gPSBwYXJzZU5vZGVOYW1lKG5hbWUpO1xuICAgICAgaWYgKCF0aGlzLmdyYXBoLm5vZGVzW25vcm1hbGl6ZWROYW1lXSkge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoYFRoZSBvdXRwdXQgJyR7bmFtZX0nIGlzIG5vdCBmb3VuZCBpbiB0aGUgZ3JhcGhgKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxufVxuIl19