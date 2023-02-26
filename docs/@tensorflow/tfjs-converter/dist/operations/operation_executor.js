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
import * as tfc from '@tensorflow/tfjs-core';
import { NodeValueImpl } from './custom_op/node_value_impl';
import { getRegisteredOp } from './custom_op/register';
import * as arithmetic from './executors/arithmetic_executor';
import * as basicMath from './executors/basic_math_executor';
import * as control from './executors/control_executor';
import * as convolution from './executors/convolution_executor';
import * as creation from './executors/creation_executor';
import * as dynamic from './executors/dynamic_executor';
import * as evaluation from './executors/evaluation_executor';
import * as graph from './executors/graph_executor';
import * as hashTable from './executors/hash_table_executor';
import * as image from './executors/image_executor';
import * as logical from './executors/logical_executor';
import * as matrices from './executors/matrices_executor';
import * as normalization from './executors/normalization_executor';
import * as ragged from './executors/ragged_executor';
import * as reduction from './executors/reduction_executor';
import * as sliceJoin from './executors/slice_join_executor';
import * as sparse from './executors/sparse_executor';
import * as spectral from './executors/spectral_executor';
import * as string from './executors/string_executor';
import * as transformation from './executors/transformation_executor';
/**
 * Executes the op defined by the node object.
 * @param node
 * @param tensorMap contains tensors for executed nodes and weights
 * @param context contains tensors and information for running the current node.
 * @param resourceManager Optional. Contains global resources of the model.
 */
export function executeOp(node, tensorMap, context, resourceManager, tidy = tfc.tidy) {
    const value = ((node, tensorMap, context) => {
        switch (node.category) {
            case 'arithmetic':
                return tidy(() => arithmetic.executeOp(node, tensorMap, context));
            case 'basic_math':
                return tidy(() => basicMath.executeOp(node, tensorMap, context));
            case 'control':
                return control.executeOp(node, tensorMap, context);
            case 'convolution':
                return tidy(() => convolution.executeOp(node, tensorMap, context));
            case 'creation':
                return tidy(() => creation.executeOp(node, tensorMap, context));
            case 'dynamic':
                return dynamic.executeOp(node, tensorMap, context);
            case 'evaluation':
                return tidy(() => evaluation.executeOp(node, tensorMap, context));
            case 'image':
                return tidy(() => image.executeOp(node, tensorMap, context));
            case 'graph':
                return tidy(() => graph.executeOp(node, tensorMap, context));
            case 'logical':
                return tidy(() => logical.executeOp(node, tensorMap, context));
            case 'matrices':
                return tidy(() => matrices.executeOp(node, tensorMap, context));
            case 'normalization':
                return tidy(() => normalization.executeOp(node, tensorMap, context));
            case 'ragged':
                return tidy(() => ragged.executeOp(node, tensorMap, context));
            case 'reduction':
                return tidy(() => reduction.executeOp(node, tensorMap, context));
            case 'slice_join':
                return tidy(() => sliceJoin.executeOp(node, tensorMap, context));
            case 'sparse':
                return tidy(() => sparse.executeOp(node, tensorMap, context));
            case 'spectral':
                return tidy(() => spectral.executeOp(node, tensorMap, context));
            case 'string':
                return tidy(() => string.executeOp(node, tensorMap, context));
            case 'transformation':
                return tidy(() => transformation.executeOp(node, tensorMap, context));
            case 'hash_table':
                return hashTable.executeOp(node, tensorMap, context, resourceManager);
            case 'custom':
                const opMapper = getRegisteredOp(node.op);
                if (opMapper && opMapper.customExecutor) {
                    return opMapper.customExecutor(new NodeValueImpl(node, tensorMap, context));
                }
                else {
                    throw TypeError(`Custom op ${node.op} is not registered.`);
                }
            default:
                throw TypeError(`Unknown op '${node.op}'. File an issue at ` +
                    `https://github.com/tensorflow/tfjs/issues so we can add it` +
                    `, or register a custom execution with tf.registerOp()`);
        }
    })(node, tensorMap, context);
    if (tfc.util.isPromise(value)) {
        return value.then((data) => [].concat(data));
    }
    return [].concat(value);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoib3BlcmF0aW9uX2V4ZWN1dG9yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb252ZXJ0ZXIvc3JjL29wZXJhdGlvbnMvb3BlcmF0aW9uX2V4ZWN1dG9yLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sS0FBSyxHQUFHLE1BQU0sdUJBQXVCLENBQUM7QUFNN0MsT0FBTyxFQUFDLGFBQWEsRUFBQyxNQUFNLDZCQUE2QixDQUFDO0FBQzFELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUNyRCxPQUFPLEtBQUssVUFBVSxNQUFNLGlDQUFpQyxDQUFDO0FBQzlELE9BQU8sS0FBSyxTQUFTLE1BQU0saUNBQWlDLENBQUM7QUFDN0QsT0FBTyxLQUFLLE9BQU8sTUFBTSw4QkFBOEIsQ0FBQztBQUN4RCxPQUFPLEtBQUssV0FBVyxNQUFNLGtDQUFrQyxDQUFDO0FBQ2hFLE9BQU8sS0FBSyxRQUFRLE1BQU0sK0JBQStCLENBQUM7QUFDMUQsT0FBTyxLQUFLLE9BQU8sTUFBTSw4QkFBOEIsQ0FBQztBQUN4RCxPQUFPLEtBQUssVUFBVSxNQUFNLGlDQUFpQyxDQUFDO0FBQzlELE9BQU8sS0FBSyxLQUFLLE1BQU0sNEJBQTRCLENBQUM7QUFDcEQsT0FBTyxLQUFLLFNBQVMsTUFBTSxpQ0FBaUMsQ0FBQztBQUM3RCxPQUFPLEtBQUssS0FBSyxNQUFNLDRCQUE0QixDQUFDO0FBQ3BELE9BQU8sS0FBSyxPQUFPLE1BQU0sOEJBQThCLENBQUM7QUFDeEQsT0FBTyxLQUFLLFFBQVEsTUFBTSwrQkFBK0IsQ0FBQztBQUMxRCxPQUFPLEtBQUssYUFBYSxNQUFNLG9DQUFvQyxDQUFDO0FBQ3BFLE9BQU8sS0FBSyxNQUFNLE1BQU0sNkJBQTZCLENBQUM7QUFDdEQsT0FBTyxLQUFLLFNBQVMsTUFBTSxnQ0FBZ0MsQ0FBQztBQUM1RCxPQUFPLEtBQUss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SSxhQUFhLENBQUMsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO2lCQUNsRDtxQkFBTTtvQkFDTCxNQUFNLFNBQVMsQ0FBQyxhQUFhLElBQUksQ0FBQyxFQUFFLHFCQUFxQixDQUFDLENBQUM7aUJBQzVEO1lBQ0g7Z0JBQ0UsTUFBTSxTQUFTLENBQ1gsZUFBZSxJQUFJLENBQUMsRUFBRSxzQkFBc0I7b0JBQzVDLDREQUE0RDtvQkFDNUQsdURBQXVELENBQUMsQ0FBQztTQUNoRTtJQUNILENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDakMsSUFBSSxHQUFHLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxLQUFLLENBQUMsRUFBRTtRQUM3QixPQUFPLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxNQUFNLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztLQUM5QztJQUNELE9BQU8sRUFBRSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQztBQUMxQixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgKiBhcyB0ZmMgZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtOYW1lZFRlbnNvcnNNYXB9IGZyb20gJy4uL2RhdGEvdHlwZXMnO1xuaW1wb3J0IHtFeGVjdXRpb25Db250ZXh0fSBmcm9tICcuLi9leGVjdXRvci9leGVjdXRpb25fY29udGV4dCc7XG5pbXBvcnQge1Jlc291cmNlTWFuYWdlcn0gZnJvbSAnLi4vZXhlY3V0b3IvcmVzb3VyY2VfbWFuYWdlcic7XG5cbmltcG9ydCB7Tm9kZVZhbHVlSW1wbH0gZnJvbSAnLi9jdXN0b21fb3Avbm9kZV92YWx1ZV9pbXBsJztcbmltcG9ydCB7Z2V0UmVnaXN0ZXJlZE9wfSBmcm9tICcuL2N1c3RvbV9vcC9yZWdpc3Rlcic7XG5pbXBvcnQgKiBhcyBhcml0aG1ldGljIGZyb20gJy4vZXhlY3V0b3JzL2FyaXRobWV0aWNfZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgYmFzaWNNYXRoIGZyb20gJy4vZXhlY3V0b3JzL2Jhc2ljX21hdGhfZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgY29udHJvbCBmcm9tICcuL2V4ZWN1dG9ycy9jb250cm9sX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIGNvbnZvbHV0aW9uIGZyb20gJy4vZXhlY3V0b3JzL2NvbnZvbHV0aW9uX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIGNyZWF0aW9uIGZyb20gJy4vZXhlY3V0b3JzL2NyZWF0aW9uX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIGR5bmFtaWMgZnJvbSAnLi9leGVjdXRvcnMvZHluYW1pY19leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBldmFsdWF0aW9uIGZyb20gJy4vZXhlY3V0b3JzL2V2YWx1YXRpb25fZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgZ3JhcGggZnJvbSAnLi9leGVjdXRvcnMvZ3JhcGhfZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgaGFzaFRhYmxlIGZyb20gJy4vZXhlY3V0b3JzL2hhc2hfdGFibGVfZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgaW1hZ2UgZnJvbSAnLi9leGVjdXRvcnMvaW1hZ2VfZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgbG9naWNhbCBmcm9tICcuL2V4ZWN1dG9ycy9sb2dpY2FsX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIG1hdHJpY2VzIGZyb20gJy4vZXhlY3V0b3JzL21hdHJpY2VzX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIG5vcm1hbGl6YXRpb24gZnJvbSAnLi9leGVjdXRvcnMvbm9ybWFsaXphdGlvbl9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyByYWdnZWQgZnJvbSAnLi9leGVjdXRvcnMvcmFnZ2VkX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIHJlZHVjdGlvbiBmcm9tICcuL2V4ZWN1dG9ycy9yZWR1Y3Rpb25fZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgc2xpY2VKb2luIGZyb20gJy4vZXhlY3V0b3JzL3NsaWNlX2pvaW5fZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgc3BhcnNlIGZyb20gJy4vZXhlY3V0b3JzL3NwYXJzZV9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBzcGVjdHJhbCBmcm9tICcuL2V4ZWN1dG9ycy9zcGVjdHJhbF9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBzdHJpbmcgZnJvbSAnLi9leGVjdXRvcnMvc3RyaW5nX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIHRyYW5zZm9ybWF0aW9uIGZyb20gJy4vZXhlY3V0b3JzL3RyYW5zZm9ybWF0aW9uX2V4ZWN1dG9yJztcbmltcG9ydCB7Tm9kZX0gZnJvbSAnLi90eXBlcyc7XG5cbi8qKlxuICogRXhlY3V0ZXMgdGhlIG9wIGRlZmluZWQgYnkgdGhlIG5vZGUgb2JqZWN0LlxuICogQHBhcmFtIG5vZGVcbiAqIEBwYXJhbSB0ZW5zb3JNYXAgY29udGFpbnMgdGVuc29ycyBmb3IgZXhlY3V0ZWQgbm9kZXMgYW5kIHdlaWdodHNcbiAqIEBwYXJhbSBjb250ZXh0IGNvbnRhaW5zIHRlbnNvcnMgYW5kIGluZm9ybWF0aW9uIGZvciBydW5uaW5nIHRoZSBjdXJyZW50IG5vZGUuXG4gKiBAcGFyYW0gcmVzb3VyY2VNYW5hZ2VyIE9wdGlvbmFsLiBDb250YWlucyBnbG9iYWwgcmVzb3VyY2VzIG9mIHRoZSBtb2RlbC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGV4ZWN1dGVPcChcbiAgICBub2RlOiBOb2RlLCB0ZW5zb3JNYXA6IE5hbWVkVGVuc29yc01hcCwgY29udGV4dDogRXhlY3V0aW9uQ29udGV4dCxcbiAgICByZXNvdXJjZU1hbmFnZXI/OiBSZXNvdXJjZU1hbmFnZXIsIHRpZHkgPSB0ZmMudGlkeSk6IHRmYy5UZW5zb3JbXXxcbiAgICBQcm9taXNlPHRmYy5UZW5zb3JbXT4ge1xuICBjb25zdCB2YWx1ZSA9XG4gICAgICAoKG5vZGU6IE5vZGUsIHRlbnNvck1hcDogTmFtZWRUZW5zb3JzTWFwLCBjb250ZXh0OiBFeGVjdXRpb25Db250ZXh0KSA9PiB7XG4gICAgICAgIHN3aXRjaCAobm9kZS5jYXRlZ29yeSkge1xuICAgICAgICAgIGNhc2UgJ2FyaXRobWV0aWMnOlxuICAgICAgICAgICAgcmV0dXJuIHRpZHkoKCkgPT4gYXJpdGhtZXRpYy5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAnYmFzaWNfbWF0aCc6XG4gICAgICAgICAgICByZXR1cm4gdGlkeSgoKSA9PiBiYXNpY01hdGguZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ2NvbnRyb2wnOlxuICAgICAgICAgICAgcmV0dXJuIGNvbnRyb2wuZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCk7XG4gICAgICAgICAgY2FzZSAnY29udm9sdXRpb24nOlxuICAgICAgICAgICAgcmV0dXJuIHRpZHkoKCkgPT4gY29udm9sdXRpb24uZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ2NyZWF0aW9uJzpcbiAgICAgICAgICAgIHJldHVybiB0aWR5KCgpID0+IGNyZWF0aW9uLmV4ZWN1dGVPcChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpKTtcbiAgICAgICAgICBjYXNlICdkeW5hbWljJzpcbiAgICAgICAgICAgIHJldHVybiBkeW5hbWljLmV4ZWN1dGVPcChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpO1xuICAgICAgICAgIGNhc2UgJ2V2YWx1YXRpb24nOlxuICAgICAgICAgICAgcmV0dXJuIHRpZHkoKCkgPT4gZXZhbHVhdGlvbi5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAnaW1hZ2UnOlxuICAgICAgICAgICAgcmV0dXJuIHRpZHkoKCkgPT4gaW1hZ2UuZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ2dyYXBoJzpcbiAgICAgICAgICAgIHJldHVybiB0aWR5KCgpID0+IGdyYXBoLmV4ZWN1dGVPcChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpKTtcbiAgICAgICAgICBjYXNlICdsb2dpY2FsJzpcbiAgICAgICAgICAgIHJldHVybiB0aWR5KCgpID0+IGxvZ2ljYWwuZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ21hdHJpY2VzJzpcbiAgICAgICAgICAgIHJldHVybiB0aWR5KCgpID0+IG1hdHJpY2VzLmV4ZWN1dGVPcChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpKTtcbiAgICAgICAgICBjYXNlICdub3JtYWxpemF0aW9uJzpcbiAgICAgICAgICAgIHJldHVybiB0aWR5KFxuICAgICAgICAgICAgICAgICgpID0+IG5vcm1hbGl6YXRpb24uZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ3JhZ2dlZCc6XG4gICAgICAgICAgICByZXR1cm4gdGlkeSgoKSA9PiByYWdnZWQuZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ3JlZHVjdGlvbic6XG4gICAgICAgICAgICByZXR1cm4gdGlkeSgoKSA9PiByZWR1Y3Rpb24uZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ3NsaWNlX2pvaW4nOlxuICAgICAgICAgICAgcmV0dXJuIHRpZHkoKCkgPT4gc2xpY2VKb2luLmV4ZWN1dGVPcChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpKTtcbiAgICAgICAgICBjYXNlICdzcGFyc2UnOlxuICAgICAgICAgICAgcmV0dXJuIHRpZHkoKCkgPT4gc3BhcnNlLmV4ZWN1dGVPcChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpKTtcbiAgICAgICAgICBjYXNlICdzcGVjdHJhbCc6XG4gICAgICAgICAgICByZXR1cm4gdGlkeSgoKSA9PiBzcGVjdHJhbC5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAnc3RyaW5nJzpcbiAgICAgICAgICAgIHJldHVybiB0aWR5KCgpID0+IHN0cmluZy5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAndHJhbnNmb3JtYXRpb24nOlxuICAgICAgICAgICAgcmV0dXJuIHRpZHkoXG4gICAgICAgICAgICAgICAgKCkgPT4gdHJhbnNmb3JtYXRpb24uZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ2hhc2hfdGFibGUnOlxuICAgICAgICAgICAgcmV0dXJuIGhhc2hUYWJsZS5leGVjdXRlT3AoXG4gICAgICAgICAgICAgICAgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0LCByZXNvdXJjZU1hbmFnZXIpO1xuICAgICAgICAgIGNhc2UgJ2N1c3RvbSc6XG4gICAgICAgICAgICBjb25zdCBvcE1hcHBlciA9IGdldFJlZ2lzdGVyZWRPcChub2RlLm9wKTtcbiAgICAgICAgICAgIGlmIChvcE1hcHBlciAmJiBvcE1hcHBlci5jdXN0b21FeGVjdXRvcikge1xuICAgICAgICAgICAgICByZXR1cm4gb3BNYXBwZXIuY3VzdG9tRXhlY3V0b3IoXG4gICAgICAgICAgICAgICAgICBuZXcgTm9kZVZhbHVlSW1wbChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIHRocm93IFR5cGVFcnJvcihgQ3VzdG9tIG9wICR7bm9kZS5vcH0gaXMgbm90IHJlZ2lzdGVyZWQuYCk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgZGVmYXVsdDpcbiAgICAgICAgICAgIHRocm93IFR5cGVFcnJvcihcbiAgICAgICAgICAgICAgICBgVW5rbm93biBvcCAnJHtub2RlLm9wfScuIEZpbGUgYW4gaXNzdWUgYXQgYCArXG4gICAgICAgICAgICAgICAgYGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzIHNvIHdlIGNhbiBhZGQgaXRgICtcbiAgICAgICAgICAgICAgICBgLCBvciByZWdpc3RlciBhIGN1c3RvbSBleGVjdXRpb24gd2l0aCB0Zi5yZWdpc3Rlck9wKClgKTtcbiAgICAgICAgfVxuICAgICAgfSkobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KTtcbiAgaWYgKHRmYy51dGlsLmlzUHJvbWlzZSh2YWx1ZSkpIHtcbiAgICByZXR1cm4gdmFsdWUudGhlbigoZGF0YSkgPT4gW10uY29uY2F0KGRhdGEpKTtcbiAgfVxuICByZXR1cm4gW10uY29uY2F0KHZhbHVlKTtcbn1cbiJdfQ==