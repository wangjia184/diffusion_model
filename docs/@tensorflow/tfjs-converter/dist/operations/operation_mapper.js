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
import { env } from '@tensorflow/tfjs-core';
import * as tensorflow from '../data/compiled_api';
import { getRegisteredOp } from './custom_op/register';
import { getNodeNameAndIndex } from './executors/utils';
import * as arithmetic from './op_list/arithmetic';
import * as basicMath from './op_list/basic_math';
import * as control from './op_list/control';
import * as convolution from './op_list/convolution';
import * as creation from './op_list/creation';
import * as dynamic from './op_list/dynamic';
import * as evaluation from './op_list/evaluation';
import * as graph from './op_list/graph';
import * as hashTable from './op_list/hash_table';
import * as image from './op_list/image';
import * as logical from './op_list/logical';
import * as matrices from './op_list/matrices';
import * as normalization from './op_list/normalization';
import * as reduction from './op_list/reduction';
import * as sliceJoin from './op_list/slice_join';
import * as sparse from './op_list/sparse';
import * as spectral from './op_list/spectral';
import * as string from './op_list/string';
import * as transformation from './op_list/transformation';
export class OperationMapper {
    // Loads the op mapping from the JSON file.
    constructor() {
        const ops = [
            arithmetic, basicMath, control, convolution, creation, dynamic,
            evaluation, graph, hashTable, image, logical, matrices, normalization,
            reduction, sliceJoin, sparse, spectral, string, transformation
        ];
        const mappersJson = [].concat(...ops.map(op => op.json));
        this.opMappers = mappersJson.reduce((map, mapper) => {
            map[mapper.tfOpName] = mapper;
            return map;
        }, {});
    }
    // Singleton instance for the mapper
    static get Instance() {
        return this._instance || (this._instance = new this());
    }
    // Converts the model inference graph from Tensorflow GraphDef to local
    // representation for TensorFlow.js API
    transformGraph(graph, signature = {}) {
        const tfNodes = graph.node;
        const placeholders = [];
        const weights = [];
        const initNodes = [];
        const nodes = tfNodes.reduce((map, node) => {
            map[node.name] = this.mapNode(node);
            if (node.op.startsWith('Placeholder')) {
                placeholders.push(map[node.name]);
            }
            else if (node.op === 'Const') {
                weights.push(map[node.name]);
            }
            else if (node.input == null || node.input.length === 0) {
                initNodes.push(map[node.name]);
            }
            return map;
        }, {});
        let inputs = [];
        const outputs = [];
        let inputNodeNameToKey = {};
        let outputNodeNameToKey = {};
        if (signature != null) {
            inputNodeNameToKey = this.mapSignatureEntries(signature.inputs);
            outputNodeNameToKey = this.mapSignatureEntries(signature.outputs);
        }
        const allNodes = Object.keys(nodes);
        allNodes.forEach(key => {
            const node = nodes[key];
            node.inputNames.forEach((name, index) => {
                const [nodeName, , outputName] = getNodeNameAndIndex(name);
                const inputNode = nodes[nodeName];
                if (inputNode.outputs != null) {
                    const outputIndex = inputNode.outputs.indexOf(outputName);
                    if (outputIndex !== -1) {
                        const inputName = `${nodeName}:${outputIndex}`;
                        // update the input name to use the mapped output index directly.
                        node.inputNames[index] = inputName;
                    }
                }
                node.inputs.push(inputNode);
                inputNode.children.push(node);
            });
        });
        // if signature has not outputs set, add any node that does not have
        // outputs.
        if (Object.keys(outputNodeNameToKey).length === 0) {
            allNodes.forEach(key => {
                const node = nodes[key];
                if (node.children.length === 0) {
                    outputs.push(node);
                }
            });
        }
        else {
            Object.keys(outputNodeNameToKey).forEach(name => {
                const [nodeName,] = getNodeNameAndIndex(name);
                const node = nodes[nodeName];
                if (node != null) {
                    node.signatureKey = outputNodeNameToKey[name];
                    outputs.push(node);
                }
            });
        }
        if (Object.keys(inputNodeNameToKey).length > 0) {
            Object.keys(inputNodeNameToKey).forEach(name => {
                const [nodeName,] = getNodeNameAndIndex(name);
                const node = nodes[nodeName];
                if (node) {
                    node.signatureKey = inputNodeNameToKey[name];
                    inputs.push(node);
                }
            });
        }
        else {
            inputs = placeholders;
        }
        let functions = {};
        if (graph.library != null && graph.library.function != null) {
            functions = graph.library.function.reduce((functions, func) => {
                functions[func.signature.name] = this.mapFunction(func);
                return functions;
            }, {});
        }
        const result = { nodes, inputs, outputs, weights, placeholders, signature, functions };
        if (initNodes.length > 0) {
            result.initNodes = initNodes;
        }
        return result;
    }
    mapSignatureEntries(entries) {
        return Object.keys(entries || {})
            .reduce((prev, curr) => {
            prev[entries[curr].name] = curr;
            return prev;
        }, {});
    }
    mapNode(node) {
        // Unsupported ops will cause an error at run-time (not parse time), since
        // they may not be used by the actual execution subgraph.
        const mapper = getRegisteredOp(node.op) || this.opMappers[node.op] || {};
        if (node.attr == null) {
            node.attr = {};
        }
        const newNode = {
            name: node.name,
            op: node.op,
            category: mapper.category,
            inputNames: (node.input ||
                []).map(input => input.startsWith('^') ? input.slice(1) : input),
            inputs: [],
            children: [],
            inputParams: {},
            attrParams: {},
            rawAttrs: node.attr,
            outputs: mapper.outputs
        };
        if (mapper.inputs != null) {
            newNode.inputParams =
                mapper.inputs.reduce((map, param) => {
                    map[param.name] = {
                        type: param.type,
                        inputIndexStart: param.start,
                        inputIndexEnd: param.end
                    };
                    return map;
                }, {});
        }
        if (mapper.attrs != null) {
            newNode.attrParams =
                mapper.attrs.reduce((map, param) => {
                    const type = param.type;
                    let value = undefined;
                    switch (param.type) {
                        case 'string':
                            value = getStringParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getStringParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'string[]':
                            value = getStringArrayParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getStringArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'number':
                            value = getNumberParam(node.attr, param.tfName, (param.defaultValue || 0));
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getNumberParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'number[]':
                            value = getNumericArrayParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getNumericArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'bool':
                            value = getBoolParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getBoolParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'bool[]':
                            value = getBoolArrayParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getBoolArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'shape':
                            value = getTensorShapeParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getTensorShapeParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'shape[]':
                            value = getTensorShapeArrayParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getTensorShapeArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'dtype':
                            value = getDtypeParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getDtypeParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'dtype[]':
                            value = getDtypeArrayParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getDtypeArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'func':
                            value = getFuncParam(node.attr, param.tfName, param.defaultValue);
                            if (value === undefined && !!param.tfDeprecatedName) {
                                value = getFuncParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                            }
                            break;
                        case 'tensor':
                        case 'tensors':
                            break;
                        default:
                            throw new Error(`Unsupported param type: ${param.type} for op: ${node.op}`);
                    }
                    map[param.name] = { value, type };
                    return map;
                }, {});
        }
        return newNode;
    }
    // map the TFunctionDef to TFJS graph object
    mapFunction(functionDef) {
        const tfNodes = functionDef.nodeDef;
        const placeholders = [];
        const weights = [];
        let nodes = {};
        if (tfNodes != null) {
            nodes = tfNodes.reduce((map, node) => {
                map[node.name] = this.mapNode(node);
                if (node.op === 'Const') {
                    weights.push(map[node.name]);
                }
                return map;
            }, {});
        }
        const inputs = [];
        const outputs = [];
        functionDef.signature.inputArg.forEach(arg => {
            const [nodeName,] = getNodeNameAndIndex(arg.name);
            const node = {
                name: nodeName,
                op: 'Placeholder',
                inputs: [],
                inputNames: [],
                category: 'graph',
                inputParams: {},
                attrParams: { dtype: { value: parseDtypeParam(arg.type), type: 'dtype' } },
                children: []
            };
            node.signatureKey = arg.name;
            inputs.push(node);
            nodes[nodeName] = node;
        });
        const allNodes = Object.keys(nodes);
        allNodes.forEach(key => {
            const node = nodes[key];
            node.inputNames.forEach((name, index) => {
                const [nodeName, , outputName] = getNodeNameAndIndex(name);
                const inputNode = nodes[nodeName];
                if (inputNode.outputs != null) {
                    const outputIndex = inputNode.outputs.indexOf(outputName);
                    if (outputIndex !== -1) {
                        const inputName = `${nodeName}:${outputIndex}`;
                        // update the input name to use the mapped output index directly.
                        node.inputNames[index] = inputName;
                    }
                }
                node.inputs.push(inputNode);
                inputNode.children.push(node);
            });
        });
        const returnNodeMap = functionDef.ret;
        functionDef.signature.outputArg.forEach(output => {
            const [nodeName, index] = getNodeNameAndIndex(returnNodeMap[output.name]);
            const node = nodes[nodeName];
            if (node != null) {
                node.defaultOutput = index;
                outputs.push(node);
            }
        });
        const signature = this.mapArgsToSignature(functionDef);
        return { nodes, inputs, outputs, weights, placeholders, signature };
    }
    mapArgsToSignature(functionDef) {
        return {
            methodName: functionDef.signature.name,
            inputs: functionDef.signature.inputArg.reduce((map, arg) => {
                map[arg.name] = this.mapArgToTensorInfo(arg);
                return map;
            }, {}),
            outputs: functionDef.signature.outputArg.reduce((map, arg) => {
                map[arg.name] = this.mapArgToTensorInfo(arg, functionDef.ret);
                return map;
            }, {}),
        };
    }
    mapArgToTensorInfo(arg, nameMap) {
        let name = arg.name;
        if (nameMap != null) {
            name = nameMap[name];
        }
        return { name, dtype: arg.type };
    }
}
export function decodeBase64(text) {
    const global = env().global;
    if (typeof global.atob !== 'undefined') {
        return global.atob(text);
    }
    else if (typeof Buffer !== 'undefined') {
        return new Buffer(text, 'base64').toString();
    }
    else {
        throw new Error('Unable to decode base64 in this environment. ' +
            'Missing built-in atob() or Buffer()');
    }
}
export function parseStringParam(s, keepCase) {
    const value = Array.isArray(s) ? String.fromCharCode.apply(null, s) : decodeBase64(s);
    return keepCase ? value : value.toLowerCase();
}
export function getStringParam(attrs, name, def, keepCase = false) {
    const param = attrs[name];
    if (param != null) {
        return parseStringParam(param.s, keepCase);
    }
    return def;
}
export function getBoolParam(attrs, name, def) {
    const param = attrs[name];
    return param ? param.b : def;
}
export function getNumberParam(attrs, name, def) {
    const param = attrs[name] || {};
    const value = param['i'] != null ? param['i'] : (param['f'] != null ? param['f'] : def);
    return (typeof value === 'number') ? value : parseInt(value, 10);
}
export function parseDtypeParam(value) {
    if (typeof (value) === 'string') {
        // tslint:disable-next-line:no-any
        value = tensorflow.DataType[value];
    }
    switch (value) {
        case tensorflow.DataType.DT_FLOAT:
        case tensorflow.DataType.DT_HALF:
            return 'float32';
        case tensorflow.DataType.DT_INT32:
        case tensorflow.DataType.DT_INT64:
        case tensorflow.DataType.DT_INT8:
        case tensorflow.DataType.DT_UINT8:
            return 'int32';
        case tensorflow.DataType.DT_BOOL:
            return 'bool';
        case tensorflow.DataType.DT_DOUBLE:
            return 'float32';
        case tensorflow.DataType.DT_STRING:
            return 'string';
        default:
            // Unknown dtype error will happen at runtime (instead of parse time),
            // since these nodes might not be used by the actual subgraph execution.
            return null;
    }
}
export function getFuncParam(attrs, name, def) {
    const param = attrs[name];
    if (param && param.func) {
        return param.func.name;
    }
    return def;
}
export function getDtypeParam(attrs, name, def) {
    const param = attrs[name];
    if (param && param.type) {
        return parseDtypeParam(param.type);
    }
    return def;
}
export function getDtypeArrayParam(attrs, name, def) {
    const param = attrs[name];
    if (param && param.list && param.list.type) {
        return param.list.type.map(v => parseDtypeParam(v));
    }
    return def;
}
export function parseTensorShapeParam(shape) {
    if (shape.unknownRank) {
        return undefined;
    }
    if (shape.dim != null) {
        return shape.dim.map(dim => (typeof dim.size === 'number') ? dim.size : parseInt(dim.size, 10));
    }
    return [];
}
export function getTensorShapeParam(attrs, name, def) {
    const param = attrs[name];
    if (param && param.shape) {
        return parseTensorShapeParam(param.shape);
    }
    return def;
}
export function getNumericArrayParam(attrs, name, def) {
    const param = attrs[name];
    if (param) {
        return ((param.list.f && param.list.f.length ? param.list.f :
            param.list.i) ||
            [])
            .map(v => (typeof v === 'number') ? v : parseInt(v, 10));
    }
    return def;
}
export function getStringArrayParam(attrs, name, def, keepCase = false) {
    const param = attrs[name];
    if (param && param.list && param.list.s) {
        return param.list.s.map((v) => {
            return parseStringParam(v, keepCase);
        });
    }
    return def;
}
export function getTensorShapeArrayParam(attrs, name, def) {
    const param = attrs[name];
    if (param && param.list && param.list.shape) {
        return param.list.shape.map((v) => {
            return parseTensorShapeParam(v);
        });
    }
    return def;
}
export function getBoolArrayParam(attrs, name, def) {
    const param = attrs[name];
    if (param && param.list && param.list.b) {
        return param.list.b;
    }
    return def;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoib3BlcmF0aW9uX21hcHBlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29udmVydGVyL3NyYy9vcGVyYXRpb25zL29wZXJhdGlvbl9tYXBwZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFXLEdBQUcsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXBELE9BQU8sS0FBSyxVQUFVLE1BQU0sc0JBQXNCLENBQUM7QUFFbkQsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLHNCQUFzQixDQUFDO0FBQ3JELE9BQU8sRUFBQyxtQkFBbUIsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQ3RELE9BQU8sS0FBSyxVQUFVLE1BQU0sc0JBQXNCLENBQUM7QUFDbkQsT0FBTyxLQUFLLFNBQVMsTUFBTSxzQkFBc0IsQ0FBQztBQUNsRCxPQUFPLEtBQUssT0FBTyxNQUFNLG1CQUFtQixDQUFDO0FBQzdDLE9BQU8sS0FBSyxXQUFXLE1BQU0sdUJBQXVCLENBQUM7QUFDckQsT0FBTyxLQUFLLFFBQVEsTUFBTSxvQkFBb0IsQ0FBQztBQUMvQyxPQUFPLEtBQUssT0FBTyxNQUFNLG1CQUFtQixDQUFDO0FBQzdDLE9BQU8sS0FBSyxVQUFVLE1BQU0sc0JBQXNCLENBQUM7QUFDbkQsT0FBTyxLQUFLLEtBQUssTUFBTSxpQkFBaUIsQ0FBQztBQUN6QyxPQUFPLEtBQUssU0FBUyxNQUFNLHNCQUFzQixDQUFDO0FBQ2xELE9BQU8sS0FBSyxLQUFLLE1BQU0saUJBQWlCLENBQUM7QUFDekMsT0FBTyxLQUFLLE9BQU8sTUFBTSxtQkFBbUIsQ0FBQztBQUM3QyxPQUFPLEtBQUssUUFBUSxNQUFNLG9CQUFvQixDQUFDO0FBQy9DLE9BQU8sS0FBSyxhQUFhLE1BQU0seUJBQXlCLENBQUM7QUFDekQsT0FBTyxLQUFLLFNBQVMsTUFBTSxxQkFBcUIsQ0FBQztBQUNqRCxPQUFPLEtB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QUMzQyxPQUFPLEtBQUssQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFO1lBQ2hDLE9BQU8scUJBQXFCLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEMsQ0FBQyxDQUFDLENBQUM7S0FDSjtJQUNELE9BQU8sR0FBRyxDQUFDO0FBQ2IsQ0FBQztBQUVELE1BQU0sVUFBVSxpQkFBaUIsQ0FDN0IsS0FBNkMsRUFBRSxJQUFZLEVBQzNELEdBQWM7SUFDaEIsTUFBTSxLQUFLLEdBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQzFCLElBQUksS0FBSyxJQUFJLEtBQUssQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUU7UUFDdkMsT0FBTyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztLQUNyQjtJQUNELE9BQU8sR0FBRyxDQUFDO0FBQ2IsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtEYXRhVHlwZSwgZW52fSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQgKiBhcyB0ZW5zb3JmbG93IGZyb20gJy4uL2RhdGEvY29tcGlsZWRfYXBpJztcblxuaW1wb3J0IHtnZXRSZWdpc3RlcmVkT3B9IGZyb20gJy4vY3VzdG9tX29wL3JlZ2lzdGVyJztcbmltcG9ydCB7Z2V0Tm9kZU5hbWVBbmRJbmRleH0gZnJvbSAnLi9leGVjdXRvcnMvdXRpbHMnO1xuaW1wb3J0ICogYXMgYXJpdGhtZXRpYyBmcm9tICcuL29wX2xpc3QvYXJpdGhtZXRpYyc7XG5pbXBvcnQgKiBhcyBiYXNpY01hdGggZnJvbSAnLi9vcF9saXN0L2Jhc2ljX21hdGgnO1xuaW1wb3J0ICogYXMgY29udHJvbCBmcm9tICcuL29wX2xpc3QvY29udHJvbCc7XG5pbXBvcnQgKiBhcyBjb252b2x1dGlvbiBmcm9tICcuL29wX2xpc3QvY29udm9sdXRpb24nO1xuaW1wb3J0ICogYXMgY3JlYXRpb24gZnJvbSAnLi9vcF9saXN0L2NyZWF0aW9uJztcbmltcG9ydCAqIGFzIGR5bmFtaWMgZnJvbSAnLi9vcF9saXN0L2R5bmFtaWMnO1xuaW1wb3J0ICogYXMgZXZhbHVhdGlvbiBmcm9tICcuL29wX2xpc3QvZXZhbHVhdGlvbic7XG5pbXBvcnQgKiBhcyBncmFwaCBmcm9tICcuL29wX2xpc3QvZ3JhcGgnO1xuaW1wb3J0ICogYXMgaGFzaFRhYmxlIGZyb20gJy4vb3BfbGlzdC9oYXNoX3RhYmxlJztcbmltcG9ydCAqIGFzIGltYWdlIGZyb20gJy4vb3BfbGlzdC9pbWFnZSc7XG5pbXBvcnQgKiBhcyBsb2dpY2FsIGZyb20gJy4vb3BfbGlzdC9sb2dpY2FsJztcbmltcG9ydCAqIGFzIG1hdHJpY2VzIGZyb20gJy4vb3BfbGlzdC9tYXRyaWNlcyc7XG5pbXBvcnQgKiBhcyBub3JtYWxpemF0aW9uIGZyb20gJy4vb3BfbGlzdC9ub3JtYWxpemF0aW9uJztcbmltcG9ydCAqIGFzIHJlZHVjdGlvbiBmcm9tICcuL29wX2xpc3QvcmVkdWN0aW9uJztcbmltcG9ydCAqIGFzIHNsaWNlSm9pbiBmcm9tICcuL29wX2xpc3Qvc2xpY2Vfam9pbic7XG5pbXBvcnQgKiBhcyBzcGFyc2UgZnJvbSAnLi9vcF9saXN0L3NwYXJzZSc7XG5pbXBvcnQgKiBhcyBzcGVjdHJhbCBmcm9tICcuL29wX2xpc3Qvc3BlY3RyYWwnO1xuaW1wb3J0ICogYXMgc3RyaW5nIGZyb20gJy4vb3BfbGlzdC9zdHJpbmcnO1xuaW1wb3J0ICogYXMgdHJhbnNmb3JtYXRpb24gZnJvbSAnLi9vcF9saXN0L3RyYW5zZm9ybWF0aW9uJztcbmltcG9ydCB7R3JhcGgsIElucHV0UGFyYW1WYWx1ZSwgTm9kZSwgT3BNYXBwZXIsIFBhcmFtVmFsdWV9IGZyb20gJy4vdHlwZXMnO1xuXG5leHBvcnQgY2xhc3MgT3BlcmF0aW9uTWFwcGVyIHtcbiAgcHJpdmF0ZSBzdGF0aWMgX2luc3RhbmNlOiBPcGVyYXRpb25NYXBwZXI7XG5cbiAgcHJpdmF0ZSBvcE1hcHBlcnM6IHtba2V5OiBzdHJpbmddOiBPcE1hcHBlcn07XG5cbiAgLy8gU2luZ2xldG9uIGluc3RhbmNlIGZvciB0aGUgbWFwcGVyXG4gIHB1YmxpYyBzdGF0aWMgZ2V0IEluc3RhbmNlKCkge1xuICAgIHJldHVybiB0aGlzLl9pbnN0YW5jZSB8fCAodGhpcy5faW5zdGFuY2UgPSBuZXcgdGhpcygpKTtcbiAgfVxuXG4gIC8vIExvYWRzIHRoZSBvcCBtYXBwaW5nIGZyb20gdGhlIEpTT04gZmlsZS5cbiAgcHJpdmF0ZSBjb25zdHJ1Y3RvcigpIHtcbiAgICBjb25zdCBvcHMgPSBbXG4gICAgICBhcml0aG1ldGljLCBiYXNpY01hdGgsIGNvbnRyb2wsIGNvbnZvbHV0aW9uLCBjcmVhdGlvbiwgZHluYW1pYyxcbiAgICAgIGV2YWx1YXRpb24sIGdyYXBoLCBoYXNoVGFibGUsIGltYWdlLCBsb2dpY2FsLCBtYXRyaWNlcywgbm9ybWFsaXphdGlvbixcbiAgICAgIHJlZHVjdGlvbiwgc2xpY2VKb2luLCBzcGFyc2UsIHNwZWN0cmFsLCBzdHJpbmcsIHRyYW5zZm9ybWF0aW9uXG4gICAgXTtcbiAgICBjb25zdCBtYXBwZXJzSnNvbjogT3BNYXBwZXJbXSA9IFtdLmNvbmNhdCguLi5vcHMubWFwKG9wID0+IG9wLmpzb24pKTtcblxuICAgIHRoaXMub3BNYXBwZXJzID0gbWFwcGVyc0pzb24ucmVkdWNlPHtba2V5OiBzdHJpbmddOiBPcE1hcHBlcn0+KFxuICAgICAgICAobWFwLCBtYXBwZXI6IE9wTWFwcGVyKSA9PiB7XG4gICAgICAgICAgbWFwW21hcHBlci50Zk9wTmFtZV0gPSBtYXBwZXI7XG4gICAgICAgICAgcmV0dXJuIG1hcDtcbiAgICAgICAgfSxcbiAgICAgICAge30pO1xuICB9XG5cbiAgLy8gQ29udmVydHMgdGhlIG1vZGVsIGluZmVyZW5jZSBncmFwaCBmcm9tIFRlbnNvcmZsb3cgR3JhcGhEZWYgdG8gbG9jYWxcbiAgLy8gcmVwcmVzZW50YXRpb24gZm9yIFRlbnNvckZsb3cuanMgQVBJXG4gIHRyYW5zZm9ybUdyYXBoKFxuICAgICAgZ3JhcGg6IHRlbnNvcmZsb3cuSUdyYXBoRGVmLFxuICAgICAgc2lnbmF0dXJlOiB0ZW5zb3JmbG93LklTaWduYXR1cmVEZWYgPSB7fSk6IEdyYXBoIHtcbiAgICBjb25zdCB0Zk5vZGVzID0gZ3JhcGgubm9kZTtcbiAgICBjb25zdCBwbGFjZWhvbGRlcnM6IE5vZGVbXSA9IFtdO1xuICAgIGNvbnN0IHdlaWdodHM6IE5vZGVbXSA9IFtdO1xuICAgIGNvbnN0IGluaXROb2RlczogTm9kZVtdID0gW107XG4gICAgY29uc3Qgbm9kZXMgPSB0Zk5vZGVzLnJlZHVjZTx7W2tleTogc3RyaW5nXTogTm9kZX0+KChtYXAsIG5vZGUpID0+IHtcbiAgICAgIG1hcFtub2RlLm5hbWVdID0gdGhpcy5tYXBOb2RlKG5vZGUpO1xuICAgICAgaWYgKG5vZGUub3Auc3RhcnRzV2l0aCgnUGxhY2Vob2xkZXInKSkge1xuICAgICAgICBwbGFjZWhvbGRlcnMucHVzaChtYXBbbm9kZS5uYW1lXSk7XG4gICAgICB9IGVsc2UgaWYgKG5vZGUub3AgPT09ICdDb25zdCcpIHtcbiAgICAgICAgd2VpZ2h0cy5wdXNoKG1hcFtub2RlLm5hbWVdKTtcbiAgICAgIH0gZWxzZSBpZiAobm9kZS5pbnB1dCA9PSBudWxsIHx8IG5vZGUuaW5wdXQubGVuZ3RoID09PSAwKSB7XG4gICAgICAgIGluaXROb2Rlcy5wdXNoKG1hcFtub2RlLm5hbWVdKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBtYXA7XG4gICAgfSwge30pO1xuXG4gICAgbGV0IGlucHV0czogTm9kZVtdID0gW107XG4gICAgY29uc3Qgb3V0cHV0czogTm9kZVtdID0gW107XG4gICAgbGV0IGlucHV0Tm9kZU5hbWVUb0tleToge1trZXk6IHN0cmluZ106IHN0cmluZ30gPSB7fTtcbiAgICBsZXQgb3V0cHV0Tm9kZU5hbWVUb0tleToge1trZXk6IHN0cmluZ106IHN0cmluZ30gPSB7fTtcbiAgICBpZiAoc2lnbmF0dXJlICE9IG51bGwpIHtcbiAgICAgIGlucHV0Tm9kZU5hbWVUb0tleSA9IHRoaXMubWFwU2lnbmF0dXJlRW50cmllcyhzaWduYXR1cmUuaW5wdXRzKTtcbiAgICAgIG91dHB1dE5vZGVOYW1lVG9LZXkgPSB0aGlzLm1hcFNpZ25hdHVyZUVudHJpZXMoc2lnbmF0dXJlLm91dHB1dHMpO1xuICAgIH1cbiAgICBjb25zdCBhbGxOb2RlcyA9IE9iamVjdC5rZXlzKG5vZGVzKTtcbiAgICBhbGxOb2Rlcy5mb3JFYWNoKGtleSA9PiB7XG4gICAgICBjb25zdCBub2RlID0gbm9kZXNba2V5XTtcbiAgICAgIG5vZGUuaW5wdXROYW1lcy5mb3JFYWNoKChuYW1lLCBpbmRleCkgPT4ge1xuICAgICAgICBjb25zdCBbbm9kZU5hbWUsICwgb3V0cHV0TmFtZV0gPSBnZXROb2RlTmFtZUFuZEluZGV4KG5hbWUpO1xuICAgICAgICBjb25zdCBpbnB1dE5vZGUgPSBub2Rlc1tub2RlTmFtZV07XG4gICAgICAgIGlmIChpbnB1dE5vZGUub3V0cHV0cyAhPSBudWxsKSB7XG4gICAgICAgICAgY29uc3Qgb3V0cHV0SW5kZXggPSBpbnB1dE5vZGUub3V0cHV0cy5pbmRleE9mKG91dHB1dE5hbWUpO1xuICAgICAgICAgIGlmIChvdXRwdXRJbmRleCAhPT0gLTEpIHtcbiAgICAgICAgICAgIGNvbnN0IGlucHV0TmFtZSA9IGAke25vZGVOYW1lfToke291dHB1dEluZGV4fWA7XG4gICAgICAgICAgICAvLyB1cGRhdGUgdGhlIGlucHV0IG5hbWUgdG8gdXNlIHRoZSBtYXBwZWQgb3V0cHV0IGluZGV4IGRpcmVjdGx5LlxuICAgICAgICAgICAgbm9kZS5pbnB1dE5hbWVzW2luZGV4XSA9IGlucHV0TmFtZTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgbm9kZS5pbnB1dHMucHVzaChpbnB1dE5vZGUpO1xuICAgICAgICBpbnB1dE5vZGUuY2hpbGRyZW4ucHVzaChub2RlKTtcbiAgICAgIH0pO1xuICAgIH0pO1xuXG4gICAgLy8gaWYgc2lnbmF0dXJlIGhhcyBub3Qgb3V0cHV0cyBzZXQsIGFkZCBhbnkgbm9kZSB0aGF0IGRvZXMgbm90IGhhdmVcbiAgICAvLyBvdXRwdXRzLlxuICAgIGlmIChPYmplY3Qua2V5cyhvdXRwdXROb2RlTmFtZVRvS2V5KS5sZW5ndGggPT09IDApIHtcbiAgICAgIGFsbE5vZGVzLmZvckVhY2goa2V5ID0+IHtcbiAgICAgICAgY29uc3Qgbm9kZSA9IG5vZGVzW2tleV07XG4gICAgICAgIGlmIChub2RlLmNoaWxkcmVuLmxlbmd0aCA9PT0gMCkge1xuICAgICAgICAgIG91dHB1dHMucHVzaChub2RlKTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIE9iamVjdC5rZXlzKG91dHB1dE5vZGVOYW1lVG9LZXkpLmZvckVhY2gobmFtZSA9PiB7XG4gICAgICAgIGNvbnN0IFtub2RlTmFtZSwgXSA9IGdldE5vZGVOYW1lQW5kSW5kZXgobmFtZSk7XG4gICAgICAgIGNvbnN0IG5vZGUgPSBub2Rlc1tub2RlTmFtZV07XG4gICAgICAgIGlmIChub2RlICE9IG51bGwpIHtcbiAgICAgICAgICBub2RlLnNpZ25hdHVyZUtleSA9IG91dHB1dE5vZGVOYW1lVG9LZXlbbmFtZV07XG4gICAgICAgICAgb3V0cHV0cy5wdXNoKG5vZGUpO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoT2JqZWN0LmtleXMoaW5wdXROb2RlTmFtZVRvS2V5KS5sZW5ndGggPiAwKSB7XG4gICAgICBPYmplY3Qua2V5cyhpbnB1dE5vZGVOYW1lVG9LZXkpLmZvckVhY2gobmFtZSA9PiB7XG4gICAgICAgIGNvbnN0IFtub2RlTmFtZSwgXSA9IGdldE5vZGVOYW1lQW5kSW5kZXgobmFtZSk7XG4gICAgICAgIGNvbnN0IG5vZGUgPSBub2Rlc1tub2RlTmFtZV07XG4gICAgICAgIGlmIChub2RlKSB7XG4gICAgICAgICAgbm9kZS5zaWduYXR1cmVLZXkgPSBpbnB1dE5vZGVOYW1lVG9LZXlbbmFtZV07XG4gICAgICAgICAgaW5wdXRzLnB1c2gobm9kZSk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH0gZWxzZSB7XG4gICAgICBpbnB1dHMgPSBwbGFjZWhvbGRlcnM7XG4gICAgfVxuXG4gICAgbGV0IGZ1bmN0aW9ucyA9IHt9O1xuICAgIGlmIChncmFwaC5saWJyYXJ5ICE9IG51bGwgJiYgZ3JhcGgubGlicmFyeS5mdW5jdGlvbiAhPSBudWxsKSB7XG4gICAgICBmdW5jdGlvbnMgPSBncmFwaC5saWJyYXJ5LmZ1bmN0aW9uLnJlZHVjZSgoZnVuY3Rpb25zLCBmdW5jKSA9PiB7XG4gICAgICAgIGZ1bmN0aW9uc1tmdW5jLnNpZ25hdHVyZS5uYW1lXSA9IHRoaXMubWFwRnVuY3Rpb24oZnVuYyk7XG4gICAgICAgIHJldHVybiBmdW5jdGlvbnM7XG4gICAgICB9LCB7fSBhcyB7W2tleTogc3RyaW5nXTogR3JhcGh9KTtcbiAgICB9XG5cbiAgICBjb25zdCByZXN1bHQ6IEdyYXBoID1cbiAgICAgICAge25vZGVzLCBpbnB1dHMsIG91dHB1dHMsIHdlaWdodHMsIHBsYWNlaG9sZGVycywgc2lnbmF0dXJlLCBmdW5jdGlvbnN9O1xuXG4gICAgaWYgKGluaXROb2Rlcy5sZW5ndGggPiAwKSB7XG4gICAgICByZXN1bHQuaW5pdE5vZGVzID0gaW5pdE5vZGVzO1xuICAgIH1cblxuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBwcml2YXRlIG1hcFNpZ25hdHVyZUVudHJpZXMoZW50cmllczoge1trOiBzdHJpbmddOiB0ZW5zb3JmbG93LklUZW5zb3JJbmZvfSkge1xuICAgIHJldHVybiBPYmplY3Qua2V5cyhlbnRyaWVzIHx8IHt9KVxuICAgICAgICAucmVkdWNlPHtba2V5OiBzdHJpbmddOiBzdHJpbmd9PigocHJldiwgY3VycikgPT4ge1xuICAgICAgICAgIHByZXZbZW50cmllc1tjdXJyXS5uYW1lXSA9IGN1cnI7XG4gICAgICAgICAgcmV0dXJuIHByZXY7XG4gICAgICAgIH0sIHt9KTtcbiAgfVxuXG4gIHByaXZhdGUgbWFwTm9kZShub2RlOiB0ZW5zb3JmbG93LklOb2RlRGVmKTogTm9kZSB7XG4gICAgLy8gVW5zdXBwb3J0ZWQgb3BzIHdpbGwgY2F1c2UgYW4gZXJyb3IgYXQgcnVuLXRpbWUgKG5vdCBwYXJzZSB0aW1lKSwgc2luY2VcbiAgICAvLyB0aGV5IG1heSBub3QgYmUgdXNlZCBieSB0aGUgYWN0dWFsIGV4ZWN1dGlvbiBzdWJncmFwaC5cbiAgICBjb25zdCBtYXBwZXIgPVxuICAgICAgICBnZXRSZWdpc3RlcmVkT3Aobm9kZS5vcCkgfHwgdGhpcy5vcE1hcHBlcnNbbm9kZS5vcF0gfHwge30gYXMgT3BNYXBwZXI7XG4gICAgaWYgKG5vZGUuYXR0ciA9PSBudWxsKSB7XG4gICAgICBub2RlLmF0dHIgPSB7fTtcbiAgICB9XG5cbiAgICBjb25zdCBuZXdOb2RlOiBOb2RlID0ge1xuICAgICAgbmFtZTogbm9kZS5uYW1lLFxuICAgICAgb3A6IG5vZGUub3AsXG4gICAgICBjYXRlZ29yeTogbWFwcGVyLmNhdGVnb3J5LFxuICAgICAgaW5wdXROYW1lczpcbiAgICAgICAgICAobm9kZS5pbnB1dCB8fFxuICAgICAgICAgICBbXSkubWFwKGlucHV0ID0+IGlucHV0LnN0YXJ0c1dpdGgoJ14nKSA/IGlucHV0LnNsaWNlKDEpIDogaW5wdXQpLFxuICAgICAgaW5wdXRzOiBbXSxcbiAgICAgIGNoaWxkcmVuOiBbXSxcbiAgICAgIGlucHV0UGFyYW1zOiB7fSxcbiAgICAgIGF0dHJQYXJhbXM6IHt9LFxuICAgICAgcmF3QXR0cnM6IG5vZGUuYXR0cixcbiAgICAgIG91dHB1dHM6IG1hcHBlci5vdXRwdXRzXG4gICAgfTtcblxuICAgIGlmIChtYXBwZXIuaW5wdXRzICE9IG51bGwpIHtcbiAgICAgIG5ld05vZGUuaW5wdXRQYXJhbXMgPVxuICAgICAgICAgIG1hcHBlci5pbnB1dHMucmVkdWNlPHtba2V5OiBzdHJpbmddOiBJbnB1dFBhcmFtVmFsdWV9PihcbiAgICAgICAgICAgICAgKG1hcCwgcGFyYW0pID0+IHtcbiAgICAgICAgICAgICAgICBtYXBbcGFyYW0ubmFtZV0gPSB7XG4gICAgICAgICAgICAgICAgICB0eXBlOiBwYXJhbS50eXBlLFxuICAgICAgICAgICAgICAgICAgaW5wdXRJbmRleFN0YXJ0OiBwYXJhbS5zdGFydCxcbiAgICAgICAgICAgICAgICAgIGlucHV0SW5kZXhFbmQ6IHBhcmFtLmVuZFxuICAgICAgICAgICAgICAgIH07XG4gICAgICAgICAgICAgICAgcmV0dXJuIG1hcDtcbiAgICAgICAgICAgICAgfSxcbiAgICAgICAgICAgICAge30pO1xuICAgIH1cbiAgICBpZiAobWFwcGVyLmF0dHJzICE9IG51bGwpIHtcbiAgICAgIG5ld05vZGUuYXR0clBhcmFtcyA9XG4gICAgICAgICAgbWFwcGVyLmF0dHJzLnJlZHVjZTx7W2tleTogc3RyaW5nXTogUGFyYW1WYWx1ZX0+KChtYXAsIHBhcmFtKSA9PiB7XG4gICAgICAgICAgICBjb25zdCB0eXBlID0gcGFyYW0udHlwZTtcbiAgICAgICAgICAgIGxldCB2YWx1ZSA9IHVuZGVmaW5lZDtcbiAgICAgICAgICAgIHN3aXRjaCAocGFyYW0udHlwZSkge1xuICAgICAgICAgICAgICBjYXNlICdzdHJpbmcnOlxuICAgICAgICAgICAgICAgIHZhbHVlID0gZ2V0U3RyaW5nUGFyYW0oXG4gICAgICAgICAgICAgICAgICAgIG5vZGUuYXR0ciwgcGFyYW0udGZOYW1lLCBwYXJhbS5kZWZhdWx0VmFsdWUgYXMgc3RyaW5nKTtcblxuICAgICAgICAgICAgICAgIGlmICh2YWx1ZSA9PT0gdW5kZWZpbmVkICYmICEhcGFyYW0udGZEZXByZWNhdGVkTmFtZSkge1xuICAgICAgICAgICAgICAgICAgdmFsdWUgPSBnZXRTdHJpbmdQYXJhbShcbiAgICAgICAgICAgICAgICAgICAgICBub2RlLmF0dHIsIHBhcmFtLnRmRGVwcmVjYXRlZE5hbWUsXG4gICAgICAgICAgICAgICAgICAgICAgcGFyYW0uZGVmYXVsdFZhbHVlIGFzIHN0cmluZyk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICBjYXNlICdzdHJpbmdbXSc6XG4gICAgICAgICAgICAgICAgdmFsdWUgPSBnZXRTdHJpbmdBcnJheVBhcmFtKFxuICAgICAgICAgICAgICAgICAgICBub2RlLmF0dHIsIHBhcmFtLnRmTmFtZSwgcGFyYW0uZGVmYXVsdFZhbHVlIGFzIHN0cmluZ1tdKTtcblxuICAgICAgICAgICAgICAgIGlmICh2YWx1ZSA9PT0gdW5kZWZpbmVkICYmICEhcGFyYW0udGZEZXByZWNhdGVkTmFtZSkge1xuICAgICAgICAgICAgICAgICAgdmFsdWUgPSBnZXRTdHJpbmdBcnJheVBhcmFtKFxuICAgICAgICAgICAgICAgICAgICAgIG5vZGUuYXR0ciwgcGFyYW0udGZEZXByZWNhdGVkTmFtZSxcbiAgICAgICAgICAgICAgICAgICAgICBwYXJhbS5kZWZhdWx0VmFsdWUgYXMgc3RyaW5nW10pO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgICAgY2FzZSAnbnVtYmVyJzpcbiAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldE51bWJlclBhcmFtKFxuICAgICAgICAgICAgICAgICAgICBub2RlLmF0dHIsIHBhcmFtLnRmTmFtZSxcbiAgICAgICAgICAgICAgICAgICAgKHBhcmFtLmRlZmF1bHRWYWx1ZSB8fCAwKSBhcyBudW1iZXIpO1xuICAgICAgICAgICAgICAgIGlmICh2YWx1ZSA9PT0gdW5kZWZpbmVkICYmICEhcGFyYW0udGZEZXByZWNhdGVkTmFtZSkge1xuICAgICAgICAgICAgICAgICAgdmFsdWUgPSBnZXROdW1iZXJQYXJhbShcbiAgICAgICAgICAgICAgICAgICAgICBub2RlLmF0dHIsIHBhcmFtLnRmRGVwcmVjYXRlZE5hbWUsXG4gICAgICAgICAgICAgICAgICAgICAgcGFyYW0uZGVmYXVsdFZhbHVlIGFzIG51bWJlcik7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICBjYXNlICdudW1iZXJbXSc6XG4gICAgICAgICAgICAgICAgdmFsdWUgPSBnZXROdW1lcmljQXJyYXlQYXJhbShcbiAgICAgICAgICAgICAgICAgICAgbm9kZS5hdHRyLCBwYXJhbS50Zk5hbWUsIHBhcmFtLmRlZmF1bHRWYWx1ZSBhcyBudW1iZXJbXSk7XG4gICAgICAgICAgICAgICAgaWYgKHZhbHVlID09PSB1bmRlZmluZWQgJiYgISFwYXJhbS50ZkRlcHJlY2F0ZWROYW1lKSB7XG4gICAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldE51bWVyaWNBcnJheVBhcmFtKFxuICAgICAgICAgICAgICAgICAgICAgIG5vZGUuYXR0ciwgcGFyYW0udGZEZXByZWNhdGVkTmFtZSxcbiAgICAgICAgICAgICAgICAgICAgICBwYXJhbS5kZWZhdWx0VmFsdWUgYXMgbnVtYmVyW10pO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgICAgY2FzZSAnYm9vbCc6XG4gICAgICAgICAgICAgICAgdmFsdWUgPSBnZXRCb29sUGFyYW0oXG4gICAgICAgICAgICAgICAgICAgIG5vZGUuYXR0ciwgcGFyYW0udGZOYW1lLCBwYXJhbS5kZWZhdWx0VmFsdWUgYXMgYm9vbGVhbik7XG4gICAgICAgICAgICAgICAgaWYgKHZhbHVlID09PSB1bmRlZmluZWQgJiYgISFwYXJhbS50ZkRlcHJlY2F0ZWROYW1lKSB7XG4gICAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldEJvb2xQYXJhbShcbiAgICAgICAgICAgICAgICAgICAgICBub2RlLmF0dHIsIHBhcmFtLnRmRGVwcmVjYXRlZE5hbWUsXG4gICAgICAgICAgICAgICAgICAgICAgcGFyYW0uZGVmYXVsdFZhbHVlIGFzIGJvb2xlYW4pO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgICAgY2FzZSAnYm9vbFtdJzpcbiAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldEJvb2xBcnJheVBhcmFtKFxuICAgICAgICAgICAgICAgICAgICBub2RlLmF0dHIsIHBhcmFtLnRmTmFtZSwgcGFyYW0uZGVmYXVsdFZhbHVlIGFzIGJvb2xlYW5bXSk7XG4gICAgICAgICAgICAgICAgaWYgKHZhbHVlID09PSB1bmRlZmluZWQgJiYgISFwYXJhbS50ZkRlcHJlY2F0ZWROYW1lKSB7XG4gICAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldEJvb2xBcnJheVBhcmFtKFxuICAgICAgICAgICAgICAgICAgICAgIG5vZGUuYXR0ciwgcGFyYW0udGZEZXByZWNhdGVkTmFtZSxcbiAgICAgICAgICAgICAgICAgICAgICBwYXJhbS5kZWZhdWx0VmFsdWUgYXMgYm9vbGVhbltdKTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICAgIGNhc2UgJ3NoYXBlJzpcbiAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldFRlbnNvclNoYXBlUGFyYW0oXG4gICAgICAgICAgICAgICAgICAgIG5vZGUuYXR0ciwgcGFyYW0udGZOYW1lLCBwYXJhbS5kZWZhdWx0VmFsdWUgYXMgbnVtYmVyW10pO1xuICAgICAgICAgICAgICAgIGlmICh2YWx1ZSA9PT0gdW5kZWZpbmVkICYmICEhcGFyYW0udGZEZXByZWNhdGVkTmFtZSkge1xuICAgICAgICAgICAgICAgICAgdmFsdWUgPSBnZXRUZW5zb3JTaGFwZVBhcmFtKFxuICAgICAgICAgICAgICAgICAgICAgIG5vZGUuYXR0ciwgcGFyYW0udGZEZXByZWNhdGVkTmFtZSxcbiAgICAgICAgICAgICAgICAgICAgICBwYXJhbS5kZWZhdWx0VmFsdWUgYXMgbnVtYmVyW10pO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgICAgY2FzZSAnc2hhcGVbXSc6XG4gICAgICAgICAgICAgICAgdmFsdWUgPSBnZXRUZW5zb3JTaGFwZUFycmF5UGFyYW0oXG4gICAgICAgICAgICAgICAgICAgIG5vZGUuYXR0ciwgcGFyYW0udGZOYW1lLCBwYXJhbS5kZWZhdWx0VmFsdWUgYXMgbnVtYmVyW11bXSk7XG4gICAgICAgICAgICAgICAgaWYgKHZhbHVlID09PSB1bmRlZmluZWQgJiYgISFwYXJhbS50ZkRlcHJlY2F0ZWROYW1lKSB7XG4gICAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldFRlbnNvclNoYXBlQXJyYXlQYXJhbShcbiAgICAgICAgICAgICAgICAgICAgICBub2RlLmF0dHIsIHBhcmFtLnRmRGVwcmVjYXRlZE5hbWUsXG4gICAgICAgICAgICAgICAgICAgICAgcGFyYW0uZGVmYXVsdFZhbHVlIGFzIG51bWJlcltdW10pO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgICAgY2FzZSAnZHR5cGUnOlxuICAgICAgICAgICAgICAgIHZhbHVlID0gZ2V0RHR5cGVQYXJhbShcbiAgICAgICAgICAgICAgICAgICAgbm9kZS5hdHRyLCBwYXJhbS50Zk5hbWUsIHBhcmFtLmRlZmF1bHRWYWx1ZSBhcyBEYXRhVHlwZSk7XG4gICAgICAgICAgICAgICAgaWYgKHZhbHVlID09PSB1bmRlZmluZWQgJiYgISFwYXJhbS50ZkRlcHJlY2F0ZWROYW1lKSB7XG4gICAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldER0eXBlUGFyYW0oXG4gICAgICAgICAgICAgICAgICAgICAgbm9kZS5hdHRyLCBwYXJhbS50ZkRlcHJlY2F0ZWROYW1lLFxuICAgICAgICAgICAgICAgICAgICAgIHBhcmFtLmRlZmF1bHRWYWx1ZSBhcyBEYXRhVHlwZSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICBjYXNlICdkdHlwZVtdJzpcbiAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldER0eXBlQXJyYXlQYXJhbShcbiAgICAgICAgICAgICAgICAgICAgbm9kZS5hdHRyLCBwYXJhbS50Zk5hbWUsIHBhcmFtLmRlZmF1bHRWYWx1ZSBhcyBEYXRhVHlwZVtdKTtcbiAgICAgICAgICAgICAgICBpZiAodmFsdWUgPT09IHVuZGVmaW5lZCAmJiAhIXBhcmFtLnRmRGVwcmVjYXRlZE5hbWUpIHtcbiAgICAgICAgICAgICAgICAgIHZhbHVlID0gZ2V0RHR5cGVBcnJheVBhcmFtKFxuICAgICAgICAgICAgICAgICAgICAgIG5vZGUuYXR0ciwgcGFyYW0udGZEZXByZWNhdGVkTmFtZSxcbiAgICAgICAgICAgICAgICAgICAgICBwYXJhbS5kZWZhdWx0VmFsdWUgYXMgRGF0YVR5cGVbXSk7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICBjYXNlICdmdW5jJzpcbiAgICAgICAgICAgICAgICB2YWx1ZSA9IGdldEZ1bmNQYXJhbShcbiAgICAgICAgICAgICAgICAgICAgbm9kZS5hdHRyLCBwYXJhbS50Zk5hbWUsIHBhcmFtLmRlZmF1bHRWYWx1ZSBhcyBzdHJpbmcpO1xuICAgICAgICAgICAgICAgIGlmICh2YWx1ZSA9PT0gdW5kZWZpbmVkICYmICEhcGFyYW0udGZEZXByZWNhdGVkTmFtZSkge1xuICAgICAgICAgICAgICAgICAgdmFsdWUgPSBnZXRGdW5jUGFyYW0oXG4gICAgICAgICAgICAgICAgICAgICAgbm9kZS5hdHRyLCBwYXJhbS50ZkRlcHJlY2F0ZWROYW1lLFxuICAgICAgICAgICAgICAgICAgICAgIHBhcmFtLmRlZmF1bHRWYWx1ZSBhcyBzdHJpbmcpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgICAgY2FzZSAndGVuc29yJzpcbiAgICAgICAgICAgICAgY2FzZSAndGVuc29ycyc6XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICAgIGRlZmF1bHQ6XG4gICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgICAgICAgICBgVW5zdXBwb3J0ZWQgcGFyYW0gdHlwZTogJHtwYXJhbS50eXBlfSBmb3Igb3A6ICR7bm9kZS5vcH1gKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIG1hcFtwYXJhbS5uYW1lXSA9IHt2YWx1ZSwgdHlwZX07XG4gICAgICAgICAgICByZXR1cm4gbWFwO1xuICAgICAgICAgIH0sIHt9KTtcbiAgICB9XG4gICAgcmV0dXJuIG5ld05vZGU7XG4gIH1cblxuICAvLyBtYXAgdGhlIFRGdW5jdGlvbkRlZiB0byBURkpTIGdyYXBoIG9iamVjdFxuICBwcml2YXRlIG1hcEZ1bmN0aW9uKGZ1bmN0aW9uRGVmOiB0ZW5zb3JmbG93LklGdW5jdGlvbkRlZik6IEdyYXBoIHtcbiAgICBjb25zdCB0Zk5vZGVzID0gZnVuY3Rpb25EZWYubm9kZURlZjtcbiAgICBjb25zdCBwbGFjZWhvbGRlcnM6IE5vZGVbXSA9IFtdO1xuICAgIGNvbnN0IHdlaWdodHM6IE5vZGVbXSA9IFtdO1xuICAgIGxldCBub2Rlczoge1trZXk6IHN0cmluZ106IE5vZGV9ID0ge307XG4gICAgaWYgKHRmTm9kZXMgIT0gbnVsbCkge1xuICAgICAgbm9kZXMgPSB0Zk5vZGVzLnJlZHVjZTx7W2tleTogc3RyaW5nXTogTm9kZX0+KChtYXAsIG5vZGUpID0+IHtcbiAgICAgICAgbWFwW25vZGUubmFtZV0gPSB0aGlzLm1hcE5vZGUobm9kZSk7XG4gICAgICAgIGlmIChub2RlLm9wID09PSAnQ29uc3QnKSB7XG4gICAgICAgICAgd2VpZ2h0cy5wdXNoKG1hcFtub2RlLm5hbWVdKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbWFwO1xuICAgICAgfSwge30pO1xuICAgIH1cbiAgICBjb25zdCBpbnB1dHM6IE5vZGVbXSA9IFtdO1xuICAgIGNvbnN0IG91dHB1dHM6IE5vZGVbXSA9IFtdO1xuXG4gICAgZnVuY3Rpb25EZWYuc2lnbmF0dXJlLmlucHV0QXJnLmZvckVhY2goYXJnID0+IHtcbiAgICAgIGNvbnN0IFtub2RlTmFtZSwgXSA9IGdldE5vZGVOYW1lQW5kSW5kZXgoYXJnLm5hbWUpO1xuICAgICAgY29uc3Qgbm9kZTogTm9kZSA9IHtcbiAgICAgICAgbmFtZTogbm9kZU5hbWUsXG4gICAgICAgIG9wOiAnUGxhY2Vob2xkZXInLFxuICAgICAgICBpbnB1dHM6IFtdLFxuICAgICAgICBpbnB1dE5hbWVzOiBbXSxcbiAgICAgICAgY2F0ZWdvcnk6ICdncmFwaCcsXG4gICAgICAgIGlucHV0UGFyYW1zOiB7fSxcbiAgICAgICAgYXR0clBhcmFtczoge2R0eXBlOiB7dmFsdWU6IHBhcnNlRHR5cGVQYXJhbShhcmcudHlwZSksIHR5cGU6ICdkdHlwZSd9fSxcbiAgICAgICAgY2hpbGRyZW46IFtdXG4gICAgICB9O1xuICAgICAgbm9kZS5zaWduYXR1cmVLZXkgPSBhcmcubmFtZTtcbiAgICAgIGlucHV0cy5wdXNoKG5vZGUpO1xuICAgICAgbm9kZXNbbm9kZU5hbWVdID0gbm9kZTtcbiAgICB9KTtcblxuICAgIGNvbnN0IGFsbE5vZGVzID0gT2JqZWN0LmtleXMobm9kZXMpO1xuICAgIGFsbE5vZGVzLmZvckVhY2goa2V5ID0+IHtcbiAgICAgIGNvbnN0IG5vZGUgPSBub2Rlc1trZXldO1xuICAgICAgbm9kZS5pbnB1dE5hbWVzLmZvckVhY2goKG5hbWUsIGluZGV4KSA9PiB7XG4gICAgICAgIGNvbnN0IFtub2RlTmFtZSwgLCBvdXRwdXROYW1lXSA9IGdldE5vZGVOYW1lQW5kSW5kZXgobmFtZSk7XG4gICAgICAgIGNvbnN0IGlucHV0Tm9kZSA9IG5vZGVzW25vZGVOYW1lXTtcbiAgICAgICAgaWYgKGlucHV0Tm9kZS5vdXRwdXRzICE9IG51bGwpIHtcbiAgICAgICAgICBjb25zdCBvdXRwdXRJbmRleCA9IGlucHV0Tm9kZS5vdXRwdXRzLmluZGV4T2Yob3V0cHV0TmFtZSk7XG4gICAgICAgICAgaWYgKG91dHB1dEluZGV4ICE9PSAtMSkge1xuICAgICAgICAgICAgY29uc3QgaW5wdXROYW1lID0gYCR7bm9kZU5hbWV9OiR7b3V0cHV0SW5kZXh9YDtcbiAgICAgICAgICAgIC8vIHVwZGF0ZSB0aGUgaW5wdXQgbmFtZSB0byB1c2UgdGhlIG1hcHBlZCBvdXRwdXQgaW5kZXggZGlyZWN0bHkuXG4gICAgICAgICAgICBub2RlLmlucHV0TmFtZXNbaW5kZXhdID0gaW5wdXROYW1lO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBub2RlLmlucHV0cy5wdXNoKGlucHV0Tm9kZSk7XG4gICAgICAgIGlucHV0Tm9kZS5jaGlsZHJlbi5wdXNoKG5vZGUpO1xuICAgICAgfSk7XG4gICAgfSk7XG5cbiAgICBjb25zdCByZXR1cm5Ob2RlTWFwID0gZnVuY3Rpb25EZWYucmV0O1xuXG4gICAgZnVuY3Rpb25EZWYuc2lnbmF0dXJlLm91dHB1dEFyZy5mb3JFYWNoKG91dHB1dCA9PiB7XG4gICAgICBjb25zdCBbbm9kZU5hbWUsIGluZGV4XSA9IGdldE5vZGVOYW1lQW5kSW5kZXgocmV0dXJuTm9kZU1hcFtvdXRwdXQubmFtZV0pO1xuICAgICAgY29uc3Qgbm9kZSA9IG5vZGVzW25vZGVOYW1lXTtcbiAgICAgIGlmIChub2RlICE9IG51bGwpIHtcbiAgICAgICAgbm9kZS5kZWZhdWx0T3V0cHV0ID0gaW5kZXg7XG4gICAgICAgIG91dHB1dHMucHVzaChub2RlKTtcbiAgICAgIH1cbiAgICB9KTtcblxuICAgIGNvbnN0IHNpZ25hdHVyZSA9IHRoaXMubWFwQXJnc1RvU2lnbmF0dXJlKGZ1bmN0aW9uRGVmKTtcbiAgICByZXR1cm4ge25vZGVzLCBpbnB1dHMsIG91dHB1dHMsIHdlaWdodHMsIHBsYWNlaG9sZGVycywgc2lnbmF0dXJlfTtcbiAgfVxuXG4gIHByaXZhdGUgbWFwQXJnc1RvU2lnbmF0dXJlKGZ1bmN0aW9uRGVmOiB0ZW5zb3JmbG93LklGdW5jdGlvbkRlZik6XG4gICAgICB0ZW5zb3JmbG93LklTaWduYXR1cmVEZWYge1xuICAgIHJldHVybiB7XG4gICAgICBtZXRob2ROYW1lOiBmdW5jdGlvbkRlZi5zaWduYXR1cmUubmFtZSxcbiAgICAgIGlucHV0czogZnVuY3Rpb25EZWYuc2lnbmF0dXJlLmlucHV0QXJnLnJlZHVjZShcbiAgICAgICAgICAobWFwLCBhcmcpID0+IHtcbiAgICAgICAgICAgIG1hcFthcmcubmFtZV0gPSB0aGlzLm1hcEFyZ1RvVGVuc29ySW5mbyhhcmcpO1xuICAgICAgICAgICAgcmV0dXJuIG1hcDtcbiAgICAgICAgICB9LFxuICAgICAgICAgIHt9IGFzIHtba2V5OiBzdHJpbmddOiB0ZW5zb3JmbG93LklUZW5zb3JJbmZvfSksXG4gICAgICBvdXRwdXRzOiBmdW5jdGlvbkRlZi5zaWduYXR1cmUub3V0cHV0QXJnLnJlZHVjZShcbiAgICAgICAgICAobWFwLCBhcmcpID0+IHtcbiAgICAgICAgICAgIG1hcFthcmcubmFtZV0gPSB0aGlzLm1hcEFyZ1RvVGVuc29ySW5mbyhhcmcsIGZ1bmN0aW9uRGVmLnJldCk7XG4gICAgICAgICAgICByZXR1cm4gbWFwO1xuICAgICAgICAgIH0sXG4gICAgICAgICAge30gYXMge1trZXk6IHN0cmluZ106IHRlbnNvcmZsb3cuSVRlbnNvckluZm99KSxcbiAgICB9O1xuICB9XG5cbiAgcHJpdmF0ZSBtYXBBcmdUb1RlbnNvckluZm8oXG4gICAgICBhcmc6IHRlbnNvcmZsb3cuT3BEZWYuSUFyZ0RlZixcbiAgICAgIG5hbWVNYXA/OiB7W2tleTogc3RyaW5nXTogc3RyaW5nfSk6IHRlbnNvcmZsb3cuSVRlbnNvckluZm8ge1xuICAgIGxldCBuYW1lID0gYXJnLm5hbWU7XG4gICAgaWYgKG5hbWVNYXAgIT0gbnVsbCkge1xuICAgICAgbmFtZSA9IG5hbWVNYXBbbmFtZV07XG4gICAgfVxuICAgIHJldHVybiB7bmFtZSwgZHR5cGU6IGFyZy50eXBlfTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZGVjb2RlQmFzZTY0KHRleHQ6IHN0cmluZyk6IHN0cmluZyB7XG4gIGNvbnN0IGdsb2JhbCA9IGVudigpLmdsb2JhbDtcbiAgaWYgKHR5cGVvZiBnbG9iYWwuYXRvYiAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICByZXR1cm4gZ2xvYmFsLmF0b2IodGV4dCk7XG4gIH0gZWxzZSBpZiAodHlwZW9mIEJ1ZmZlciAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICByZXR1cm4gbmV3IEJ1ZmZlcih0ZXh0LCAnYmFzZTY0JykudG9TdHJpbmcoKTtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdVbmFibGUgdG8gZGVjb2RlIGJhc2U2NCBpbiB0aGlzIGVudmlyb25tZW50LiAnICtcbiAgICAgICAgJ01pc3NpbmcgYnVpbHQtaW4gYXRvYigpIG9yIEJ1ZmZlcigpJyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHBhcnNlU3RyaW5nUGFyYW0oczogW118c3RyaW5nLCBrZWVwQ2FzZTogYm9vbGVhbik6IHN0cmluZyB7XG4gIGNvbnN0IHZhbHVlID1cbiAgICAgIEFycmF5LmlzQXJyYXkocykgPyBTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIHMpIDogZGVjb2RlQmFzZTY0KHMpO1xuICByZXR1cm4ga2VlcENhc2UgPyB2YWx1ZSA6IHZhbHVlLnRvTG93ZXJDYXNlKCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRTdHJpbmdQYXJhbShcbiAgICBhdHRyczoge1trZXk6IHN0cmluZ106IHRlbnNvcmZsb3cuSUF0dHJWYWx1ZX0sIG5hbWU6IHN0cmluZywgZGVmOiBzdHJpbmcsXG4gICAga2VlcENhc2UgPSBmYWxzZSk6IHN0cmluZyB7XG4gIGNvbnN0IHBhcmFtID0gYXR0cnNbbmFtZV07XG4gIGlmIChwYXJhbSAhPSBudWxsKSB7XG4gICAgcmV0dXJuIHBhcnNlU3RyaW5nUGFyYW0ocGFyYW0ucywga2VlcENhc2UpO1xuICB9XG4gIHJldHVybiBkZWY7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRCb29sUGFyYW0oXG4gICAgYXR0cnM6IHtba2V5OiBzdHJpbmddOiB0ZW5zb3JmbG93LklBdHRyVmFsdWV9LCBuYW1lOiBzdHJpbmcsXG4gICAgZGVmOiBib29sZWFuKTogYm9vbGVhbiB7XG4gIGNvbnN0IHBhcmFtID0gYXR0cnNbbmFtZV07XG4gIHJldHVybiBwYXJhbSA/IHBhcmFtLmIgOiBkZWY7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXROdW1iZXJQYXJhbShcbiAgICBhdHRyczoge1trZXk6IHN0cmluZ106IHRlbnNvcmZsb3cuSUF0dHJWYWx1ZX0sIG5hbWU6IHN0cmluZyxcbiAgICBkZWY6IG51bWJlcik6IG51bWJlciB7XG4gIGNvbnN0IHBhcmFtID0gYXR0cnNbbmFtZV0gfHwge307XG4gIGNvbnN0IHZhbHVlID1cbiAgICAgIHBhcmFtWydpJ10gIT0gbnVsbCA/IHBhcmFtWydpJ10gOiAocGFyYW1bJ2YnXSAhPSBudWxsID8gcGFyYW1bJ2YnXSA6IGRlZik7XG4gIHJldHVybiAodHlwZW9mIHZhbHVlID09PSAnbnVtYmVyJykgPyB2YWx1ZSA6IHBhcnNlSW50KHZhbHVlLCAxMCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBwYXJzZUR0eXBlUGFyYW0odmFsdWU6IHN0cmluZ3x0ZW5zb3JmbG93LkRhdGFUeXBlKTogRGF0YVR5cGUge1xuICBpZiAodHlwZW9mICh2YWx1ZSkgPT09ICdzdHJpbmcnKSB7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIHZhbHVlID0gdGVuc29yZmxvdy5EYXRhVHlwZVt2YWx1ZSBhcyBhbnldO1xuICB9XG4gIHN3aXRjaCAodmFsdWUpIHtcbiAgICBjYXNlIHRlbnNvcmZsb3cuRGF0YVR5cGUuRFRfRkxPQVQ6XG4gICAgY2FzZSB0ZW5zb3JmbG93LkRhdGFUeXBlLkRUX0hBTEY6XG4gICAgICByZXR1cm4gJ2Zsb2F0MzInO1xuICAgIGNhc2UgdGVuc29yZmxvdy5EYXRhVHlwZS5EVF9JTlQzMjpcbiAgICBjYXNlIHRlbnNvcmZsb3cuRGF0YVR5cGUuRFRfSU5UNjQ6XG4gICAgY2FzZSB0ZW5zb3JmbG93LkRhdGFUeXBlLkRUX0lOVDg6XG4gICAgY2FzZSB0ZW5zb3JmbG93LkRhdGFUeXBlLkRUX1VJTlQ4OlxuICAgICAgcmV0dXJuICdpbnQzMic7XG4gICAgY2FzZSB0ZW5zb3JmbG93LkRhdGFUeXBlLkRUX0JPT0w6XG4gICAgICByZXR1cm4gJ2Jvb2wnO1xuICAgIGNhc2UgdGVuc29yZmxvdy5EYXRhVHlwZS5EVF9ET1VCTEU6XG4gICAgICByZXR1cm4gJ2Zsb2F0MzInO1xuICAgIGNhc2UgdGVuc29yZmxvdy5EYXRhVHlwZS5EVF9TVFJJTkc6XG4gICAgICByZXR1cm4gJ3N0cmluZyc7XG4gICAgZGVmYXVsdDpcbiAgICAgIC8vIFVua25vd24gZHR5cGUgZXJyb3Igd2lsbCBoYXBwZW4gYXQgcnVudGltZSAoaW5zdGVhZCBvZiBwYXJzZSB0aW1lKSxcbiAgICAgIC8vIHNpbmNlIHRoZXNlIG5vZGVzIG1pZ2h0IG5vdCBiZSB1c2VkIGJ5IHRoZSBhY3R1YWwgc3ViZ3JhcGggZXhlY3V0aW9uLlxuICAgICAgcmV0dXJuIG51bGw7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldEZ1bmNQYXJhbShcbiAgICBhdHRyczoge1trZXk6IHN0cmluZ106IHRlbnNvcmZsb3cuSUF0dHJWYWx1ZX0sIG5hbWU6IHN0cmluZyxcbiAgICBkZWY6IHN0cmluZyk6IHN0cmluZyB7XG4gIGNvbnN0IHBhcmFtID0gYXR0cnNbbmFtZV07XG4gIGlmIChwYXJhbSAmJiBwYXJhbS5mdW5jKSB7XG4gICAgcmV0dXJuIHBhcmFtLmZ1bmMubmFtZTtcbiAgfVxuICByZXR1cm4gZGVmO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RHR5cGVQYXJhbShcbiAgICBhdHRyczoge1trZXk6IHN0cmluZ106IHRlbnNvcmZsb3cuSUF0dHJWYWx1ZX0sIG5hbWU6IHN0cmluZyxcbiAgICBkZWY6IERhdGFUeXBlKTogRGF0YVR5cGUge1xuICBjb25zdCBwYXJhbSA9IGF0dHJzW25hbWVdO1xuICBpZiAocGFyYW0gJiYgcGFyYW0udHlwZSkge1xuICAgIHJldHVybiBwYXJzZUR0eXBlUGFyYW0ocGFyYW0udHlwZSk7XG4gIH1cbiAgcmV0dXJuIGRlZjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldER0eXBlQXJyYXlQYXJhbShcbiAgICBhdHRyczoge1trZXk6IHN0cmluZ106IHRlbnNvcmZsb3cuSUF0dHJWYWx1ZX0sIG5hbWU6IHN0cmluZyxcbiAgICBkZWY6IERhdGFUeXBlW10pOiBEYXRhVHlwZVtdIHtcbiAgY29uc3QgcGFyYW0gPSBhdHRyc1tuYW1lXTtcbiAgaWYgKHBhcmFtICYmIHBhcmFtLmxpc3QgJiYgcGFyYW0ubGlzdC50eXBlKSB7XG4gICAgcmV0dXJuIHBhcmFtLmxpc3QudHlwZS5tYXAodiA9PiBwYXJzZUR0eXBlUGFyYW0odikpO1xuICB9XG4gIHJldHVybiBkZWY7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBwYXJzZVRlbnNvclNoYXBlUGFyYW0oc2hhcGU6IHRlbnNvcmZsb3cuSVRlbnNvclNoYXBlKTogbnVtYmVyW118XG4gICAgdW5kZWZpbmVkIHtcbiAgaWYgKHNoYXBlLnVua25vd25SYW5rKSB7XG4gICAgcmV0dXJuIHVuZGVmaW5lZDtcbiAgfVxuICBpZiAoc2hhcGUuZGltICE9IG51bGwpIHtcbiAgICByZXR1cm4gc2hhcGUuZGltLm1hcChcbiAgICAgICAgZGltID0+XG4gICAgICAgICAgICAodHlwZW9mIGRpbS5zaXplID09PSAnbnVtYmVyJykgPyBkaW0uc2l6ZSA6IHBhcnNlSW50KGRpbS5zaXplLCAxMCkpO1xuICB9XG4gIHJldHVybiBbXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFRlbnNvclNoYXBlUGFyYW0oXG4gICAgYXR0cnM6IHtba2V5OiBzdHJpbmddOiB0ZW5zb3JmbG93LklBdHRyVmFsdWV9LCBuYW1lOiBzdHJpbmcsXG4gICAgZGVmPzogbnVtYmVyW10pOiBudW1iZXJbXXx1bmRlZmluZWQge1xuICBjb25zdCBwYXJhbSA9IGF0dHJzW25hbWVdO1xuICBpZiAocGFyYW0gJiYgcGFyYW0uc2hhcGUpIHtcbiAgICByZXR1cm4gcGFyc2VUZW5zb3JTaGFwZVBhcmFtKHBhcmFtLnNoYXBlKTtcbiAgfVxuICByZXR1cm4gZGVmO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0TnVtZXJpY0FycmF5UGFyYW0oXG4gICAgYXR0cnM6IHtba2V5OiBzdHJpbmddOiB0ZW5zb3JmbG93LklBdHRyVmFsdWV9LCBuYW1lOiBzdHJpbmcsXG4gICAgZGVmOiBudW1iZXJbXSk6IG51bWJlcltdIHtcbiAgY29uc3QgcGFyYW0gPSBhdHRyc1tuYW1lXTtcbiAgaWYgKHBhcmFtKSB7XG4gICAgcmV0dXJuICgocGFyYW0ubGlzdC5mICYmIHBhcmFtLmxpc3QuZi5sZW5ndGggPyBwYXJhbS5saXN0LmYgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcGFyYW0ubGlzdC5pKSB8fFxuICAgICAgICAgICAgW10pXG4gICAgICAgIC5tYXAodiA9PiAodHlwZW9mIHYgPT09ICdudW1iZXInKSA/IHYgOiBwYXJzZUludCh2LCAxMCkpO1xuICB9XG4gIHJldHVybiBkZWY7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRTdHJpbmdBcnJheVBhcmFtKFxuICAgIGF0dHJzOiB7W2tleTogc3RyaW5nXTogdGVuc29yZmxvdy5JQXR0clZhbHVlfSwgbmFtZTogc3RyaW5nLCBkZWY6IHN0cmluZ1tdLFxuICAgIGtlZXBDYXNlID0gZmFsc2UpOiBzdHJpbmdbXSB7XG4gIGNvbnN0IHBhcmFtID0gYXR0cnNbbmFtZV07XG4gIGlmIChwYXJhbSAmJiBwYXJhbS5saXN0ICYmIHBhcmFtLmxpc3Qucykge1xuICAgIHJldHVybiBwYXJhbS5saXN0LnMubWFwKCh2KSA9PiB7XG4gICAgICByZXR1cm4gcGFyc2VTdHJpbmdQYXJhbSh2LCBrZWVwQ2FzZSk7XG4gICAgfSk7XG4gIH1cbiAgcmV0dXJuIGRlZjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFRlbnNvclNoYXBlQXJyYXlQYXJhbShcbiAgICBhdHRyczoge1trZXk6IHN0cmluZ106IHRlbnNvcmZsb3cuSUF0dHJWYWx1ZX0sIG5hbWU6IHN0cmluZyxcbiAgICBkZWY6IG51bWJlcltdW10pOiBudW1iZXJbXVtdIHtcbiAgY29uc3QgcGFyYW0gPSBhdHRyc1tuYW1lXTtcbiAgaWYgKHBhcmFtICYmIHBhcmFtLmxpc3QgJiYgcGFyYW0ubGlzdC5zaGFwZSkge1xuICAgIHJldHVybiBwYXJhbS5saXN0LnNoYXBlLm1hcCgodikgPT4ge1xuICAgICAgcmV0dXJuIHBhcnNlVGVuc29yU2hhcGVQYXJhbSh2KTtcbiAgICB9KTtcbiAgfVxuICByZXR1cm4gZGVmO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0Qm9vbEFycmF5UGFyYW0oXG4gICAgYXR0cnM6IHtba2V5OiBzdHJpbmddOiB0ZW5zb3JmbG93LklBdHRyVmFsdWV9LCBuYW1lOiBzdHJpbmcsXG4gICAgZGVmOiBib29sZWFuW10pOiBib29sZWFuW10ge1xuICBjb25zdCBwYXJhbSA9IGF0dHJzW25hbWVdO1xuICBpZiAocGFyYW0gJiYgcGFyYW0ubGlzdCAmJiBwYXJhbS5saXN0LmIpIHtcbiAgICByZXR1cm4gcGFyYW0ubGlzdC5iO1xuICB9XG4gIHJldHVybiBkZWY7XG59XG4iXX0=