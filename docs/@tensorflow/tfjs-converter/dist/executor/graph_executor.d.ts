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
/// <amd-module name="@tensorflow/tfjs-converter/dist/executor/graph_executor" />
import { NamedTensorMap, Tensor } from '@tensorflow/tfjs-core';
import { ISignatureDef } from '../data/compiled_api';
import { NamedTensorsMap, TensorArrayMap, TensorInfo, TensorListMap } from '../data/types';
import { Graph } from '../operations/types';
import { ResourceManager } from './resource_manager';
import { FunctionExecutor } from './types';
export declare class GraphExecutor implements FunctionExecutor {
    private graph;
    private parent?;
    private compiledMap;
    private _weightMap;
    private _weightIds;
    private _signature;
    private _inputs;
    private _outputs;
    private _initNodes;
    private SEPERATOR;
    private _functions;
    private _functionExecutorMap;
    private _resourceManager;
    private clonedTensorsMap;
    private keepIntermediateTensors;
    get weightIds(): number[];
    get functionExecutorMap(): {
        [key: string]: FunctionExecutor;
    };
    get weightMap(): NamedTensorsMap;
    set weightMap(weightMap: NamedTensorsMap);
    /**
     * Set `ResourceManager` shared by executors of a model.
     * @param resourceManager: `ResourceManager` of the `GraphModel`.
     */
    set resourceManager(resourceManager: ResourceManager);
    get inputs(): TensorInfo[];
    get outputs(): TensorInfo[];
    get inputNodes(): string[];
    get outputNodes(): string[];
    get functions(): {
        [key: string]: ISignatureDef;
    };
    /**
     *
     * @param graph Graph the model or function graph to be executed.
     * @param parent When building function exector you need to set the parent
     * executor. Since the weights and function executor maps are set at parant
     * level, that function executor can access the function maps and weight maps
     * through the parent.
     */
    constructor(graph: Graph, parent?: GraphExecutor);
    private getCompilationKey;
    /**
     * Compiles the inference graph and returns the minimal set of nodes that are
     * required for execution, in the correct execution order.
     */
    private compile;
    private cloneAndKeepTensor;
    private cloneTensorList;
    private cloneTensorMap;
    /**
     * Executes the inference for given input tensors.
     * @param inputs Tensor map for the model inputs, keyed by the input node
     * names.
     * @param outputs Optional. output node name from the Tensorflow model, if
     * no outputs are specified, the default outputs of the model would be used.
     * You can inspect intermediate nodes of the model by adding them to the
     * outputs array.
     */
    execute(inputs: NamedTensorMap, outputs?: string[]): Tensor[];
    private getFrozenTensorIds;
    private checkTensorForDisposal;
    /**
     * Executes the inference for given input tensors in Async fashion.
     * @param inputs Tensor map for the model inputs, keyed by the input node
     * names.
     * @param outputs output node name from the Tensorflow model, if no outputs
     * are specified, the default outputs of the model would be used. You can
     * inspect intermediate nodes of the model by adding them to the outputs
     * array.
     */
    executeAsync(inputs: NamedTensorMap, outputs?: string[]): Promise<Tensor[]>;
    disposeIntermediateTensors(): void;
    getIntermediateTensors(): NamedTensorsMap;
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
    private _executeAsync;
    executeFunctionAsync(inputs: Tensor[], tensorArrayMap: TensorArrayMap, tensorListMap: TensorListMap): Promise<Tensor[]>;
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
    private executeWithControlFlow;
    private processStack;
    private processChildNodes;
    /**
     * Releases the memory used by the weight tensors.
     */
    dispose(): void;
    private checkInputShapeAndType;
    private mapInputs;
    private checkInputs;
    private mapOutputs;
    private checkOutputs;
}
