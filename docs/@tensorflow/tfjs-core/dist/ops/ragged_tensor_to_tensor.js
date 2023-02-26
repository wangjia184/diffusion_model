/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import { ENGINE } from '../engine';
import { RaggedTensorToTensor } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Create a dense tensor from a ragged tensor, possibly altering its shape.
 *
 * The raggedTensorToTensor op creates a dense tensor from am array of row
 * partition tensors, a value vector, and default values. If the shape is
 * unspecified, the minimal shape required to contain all the elements in the
 * ragged tensor (the natural shape) will be used. If some dimensions are left
 * unspecified, then the size of the natural shape is used in that dimension.
 *
 * The defaultValue will be broadcast to the output shape. After that, the
 * values from the ragged tensor overwrite the default values. Note that the
 * defaultValue must have less dimensions than the value.
 *
 * The row partition tensors are in the order of the dimensions. At present, the
 * types can be: "ROW_SPLITS": the row_splits tensor from the ragged tensor.
 *   "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
 *   "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then it
 * is preceded by "FIRST_DIM_SIZE".
 * ```
 * @param shape: A Tensor. Must be one of the following types: 'int32'. The
 *     desired shape of the output tensor. If left unspecified (empty), the
 *     minimal shape required to contain all the elements in the ragged tensor
 *     (the natural shape) will be used. If some dimensions are left
 *     unspecified, then the size of the natural shape is used in that
 *     dimension.
 *
 *     Note that dense dimensions cannot be modified by the shape argument.
 *     Trying to change the size of a dense dimension will cause the op to fail.
 *     Examples: natural shape: [4, 5, 6] shape: -1 output shape: [4, 5, 6]
 *
 *     natural shape: [4, 5, 6] shape: [3, -1, 2] output shape: [3, 5, 2]
 *
 *     natural shape: [4, 5, 6] shape: [3, 7, 2] output shape: [3, 7, 2]
 * @param values: A Tensor. A 1D tensor representing the values of the ragged
 *     tensor.
 * @param defaultValue: A Tensor. Must have the same type as values. The
 *     defaultValue when the shape is larger than the ragged tensor. The
 *     defaultValue is broadcast until it is the shape of the output tensor,
 *     and then overwritten by values in the ragged tensor. The default value
 *     must be compatible with this broadcast operation, and must have fewer
 *     dimensions than the value tensor.
 * @param rowPartitionTensors: A list of at least 1 Tensor objects with the same
 *     type in: 'int32'.
 * @param rowPartitionTypes: A list of strings. The types of the row partition
 *     tensors. At present, these can be:
 *     "ROW_SPLITS": the row_splits tensor from the ragged tensor.
 *     "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
 *     "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then
 *         it is preceeded by "FIRST_DIM_SIZE". The tensors are in the order of
 *         the dimensions.
 * @return A Tensor. Has the same type as values.
 * @doc {heading: 'Operations', subheading: 'Ragged'}
 */
function raggedTensorToTensor_(shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes) {
    const $shape = convertToTensor(shape, 'shape', 'raggedTensorToTensor', 'int32');
    const $values = convertToTensor(values, 'values', 'raggedTensorToTensor');
    const $defaultValue = convertToTensor(defaultValue, 'defaultValue', 'raggedTensorToTensor', $values.dtype);
    const $rowPartitionTensors = rowPartitionTensors.map((t, i) => convertToTensor(t, `tensors${i}`, 'raggedTensorToTensor', 'int32'));
    const inputs = {
        shape: $shape,
        values: $values,
        defaultValue: $defaultValue,
        rowPartitionTensors: $rowPartitionTensors
    };
    const attrs = { rowPartitionTypes };
    return ENGINE.runKernel(RaggedTensorToTensor, inputs, attrs);
}
export const raggedTensorToTensor = /* @__PURE__ */ op({ raggedTensorToTensor_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFnZ2VkX3RlbnNvcl90b190ZW5zb3IuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9yYWdnZWRfdGVuc29yX3RvX3RlbnNvci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxvQkFBb0IsRUFBd0QsTUFBTSxpQkFBaUIsQ0FBQztBQUU1RyxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQW9ERztBQUNILFNBQVMscUJBQXFCLENBQzFCLEtBQXdCLEVBQUUsTUFBeUIsRUFDbkQsWUFBK0IsRUFBRSxtQkFBNkIsRUFDOUQsaUJBQTJCO0lBQzdCLE1BQU0sTUFBTSxHQUNSLGVBQWUsQ0FBQyxLQUFLLEVBQUUsT0FBTyxFQUFFLHNCQUFzQixFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ3JFLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLHNCQUFzQixDQUFDLENBQUM7SUFDMUUsTUFBTSxhQUFhLEdBQUcsZUFBZSxDQUNqQyxZQUFZLEVBQUUsY0FBYyxFQUFFLHNCQUFzQixFQUFFLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUN6RSxNQUFNLG9CQUFvQixHQUFHLG1CQUFtQixDQUFDLEdBQUcsQ0FDaEQsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FDTCxlQUFlLENBQUMsQ0FBQyxFQUFFLFVBQVUsQ0FBQyxFQUFFLEVBQUUsc0JBQXNCLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUU1RSxNQUFNLE1BQU0sR0FBK0I7UUFDekMsS0FBSyxFQUFFLE1BQU07UUFDYixNQUFNLEVBQUUsT0FBTztRQUNmLFlBQVksRUFBRSxhQUFhO1FBQzNCLG1CQUFtQixFQUFFLG9CQUFvQjtLQUMxQyxDQUFDO0lBQ0YsTUFBTSxLQUFLLEdBQThCLEVBQUMsaUJBQWlCLEVBQUMsQ0FBQztJQUU3RCxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQUMsb0JBQW9CLEVBQUUsTUFBWSxFQUFFLEtBQVcsQ0FBQyxDQUFDO0FBQzNFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxvQkFBb0IsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMscUJBQXFCLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7UmFnZ2VkVGVuc29yVG9UZW5zb3IsIFJhZ2dlZFRlbnNvclRvVGVuc29yQXR0cnMsIFJhZ2dlZFRlbnNvclRvVGVuc29ySW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIENyZWF0ZSBhIGRlbnNlIHRlbnNvciBmcm9tIGEgcmFnZ2VkIHRlbnNvciwgcG9zc2libHkgYWx0ZXJpbmcgaXRzIHNoYXBlLlxuICpcbiAqIFRoZSByYWdnZWRUZW5zb3JUb1RlbnNvciBvcCBjcmVhdGVzIGEgZGVuc2UgdGVuc29yIGZyb20gYW0gYXJyYXkgb2Ygcm93XG4gKiBwYXJ0aXRpb24gdGVuc29ycywgYSB2YWx1ZSB2ZWN0b3IsIGFuZCBkZWZhdWx0IHZhbHVlcy4gSWYgdGhlIHNoYXBlIGlzXG4gKiB1bnNwZWNpZmllZCwgdGhlIG1pbmltYWwgc2hhcGUgcmVxdWlyZWQgdG8gY29udGFpbiBhbGwgdGhlIGVsZW1lbnRzIGluIHRoZVxuICogcmFnZ2VkIHRlbnNvciAodGhlIG5hdHVyYWwgc2hhcGUpIHdpbGwgYmUgdXNlZC4gSWYgc29tZSBkaW1lbnNpb25zIGFyZSBsZWZ0XG4gKiB1bnNwZWNpZmllZCwgdGhlbiB0aGUgc2l6ZSBvZiB0aGUgbmF0dXJhbCBzaGFwZSBpcyB1c2VkIGluIHRoYXQgZGltZW5zaW9uLlxuICpcbiAqIFRoZSBkZWZhdWx0VmFsdWUgd2lsbCBiZSBicm9hZGNhc3QgdG8gdGhlIG91dHB1dCBzaGFwZS4gQWZ0ZXIgdGhhdCwgdGhlXG4gKiB2YWx1ZXMgZnJvbSB0aGUgcmFnZ2VkIHRlbnNvciBvdmVyd3JpdGUgdGhlIGRlZmF1bHQgdmFsdWVzLiBOb3RlIHRoYXQgdGhlXG4gKiBkZWZhdWx0VmFsdWUgbXVzdCBoYXZlIGxlc3MgZGltZW5zaW9ucyB0aGFuIHRoZSB2YWx1ZS5cbiAqXG4gKiBUaGUgcm93IHBhcnRpdGlvbiB0ZW5zb3JzIGFyZSBpbiB0aGUgb3JkZXIgb2YgdGhlIGRpbWVuc2lvbnMuIEF0IHByZXNlbnQsIHRoZVxuICogdHlwZXMgY2FuIGJlOiBcIlJPV19TUExJVFNcIjogdGhlIHJvd19zcGxpdHMgdGVuc29yIGZyb20gdGhlIHJhZ2dlZCB0ZW5zb3IuXG4gKiAgIFwiVkFMVUVfUk9XSURTXCI6IHRoZSB2YWx1ZV9yb3dpZHMgdGVuc29yIGZyb20gdGhlIHJhZ2dlZCB0ZW5zb3IuXG4gKiAgIFwiRklSU1RfRElNX1NJWkVcIjogaWYgdmFsdWVfcm93aWRzIGlzIHVzZWQgZm9yIHRoZSBmaXJzdCBkaW1lbnNpb24sIHRoZW4gaXRcbiAqIGlzIHByZWNlZGVkIGJ5IFwiRklSU1RfRElNX1NJWkVcIi5cbiAqIGBgYFxuICogQHBhcmFtIHNoYXBlOiBBIFRlbnNvci4gTXVzdCBiZSBvbmUgb2YgdGhlIGZvbGxvd2luZyB0eXBlczogJ2ludDMyJy4gVGhlXG4gKiAgICAgZGVzaXJlZCBzaGFwZSBvZiB0aGUgb3V0cHV0IHRlbnNvci4gSWYgbGVmdCB1bnNwZWNpZmllZCAoZW1wdHkpLCB0aGVcbiAqICAgICBtaW5pbWFsIHNoYXBlIHJlcXVpcmVkIHRvIGNvbnRhaW4gYWxsIHRoZSBlbGVtZW50cyBpbiB0aGUgcmFnZ2VkIHRlbnNvclxuICogICAgICh0aGUgbmF0dXJhbCBzaGFwZSkgd2lsbCBiZSB1c2VkLiBJZiBzb21lIGRpbWVuc2lvbnMgYXJlIGxlZnRcbiAqICAgICB1bnNwZWNpZmllZCwgdGhlbiB0aGUgc2l6ZSBvZiB0aGUgbmF0dXJhbCBzaGFwZSBpcyB1c2VkIGluIHRoYXRcbiAqICAgICBkaW1lbnNpb24uXG4gKlxuICogICAgIE5vdGUgdGhhdCBkZW5zZSBkaW1lbnNpb25zIGNhbm5vdCBiZSBtb2RpZmllZCBieSB0aGUgc2hhcGUgYXJndW1lbnQuXG4gKiAgICAgVHJ5aW5nIHRvIGNoYW5nZSB0aGUgc2l6ZSBvZiBhIGRlbnNlIGRpbWVuc2lvbiB3aWxsIGNhdXNlIHRoZSBvcCB0byBmYWlsLlxuICogICAgIEV4YW1wbGVzOiBuYXR1cmFsIHNoYXBlOiBbNCwgNSwgNl0gc2hhcGU6IC0xIG91dHB1dCBzaGFwZTogWzQsIDUsIDZdXG4gKlxuICogICAgIG5hdHVyYWwgc2hhcGU6IFs0LCA1LCA2XSBzaGFwZTogWzMsIC0xLCAyXSBvdXRwdXQgc2hhcGU6IFszLCA1LCAyXVxuICpcbiAqICAgICBuYXR1cmFsIHNoYXBlOiBbNCwgNSwgNl0gc2hhcGU6IFszLCA3LCAyXSBvdXRwdXQgc2hhcGU6IFszLCA3LCAyXVxuICogQHBhcmFtIHZhbHVlczogQSBUZW5zb3IuIEEgMUQgdGVuc29yIHJlcHJlc2VudGluZyB0aGUgdmFsdWVzIG9mIHRoZSByYWdnZWRcbiAqICAgICB0ZW5zb3IuXG4gKiBAcGFyYW0gZGVmYXVsdFZhbHVlOiBBIFRlbnNvci4gTXVzdCBoYXZlIHRoZSBzYW1lIHR5cGUgYXMgdmFsdWVzLiBUaGVcbiAqICAgICBkZWZhdWx0VmFsdWUgd2hlbiB0aGUgc2hhcGUgaXMgbGFyZ2VyIHRoYW4gdGhlIHJhZ2dlZCB0ZW5zb3IuIFRoZVxuICogICAgIGRlZmF1bHRWYWx1ZSBpcyBicm9hZGNhc3QgdW50aWwgaXQgaXMgdGhlIHNoYXBlIG9mIHRoZSBvdXRwdXQgdGVuc29yLFxuICogICAgIGFuZCB0aGVuIG92ZXJ3cml0dGVuIGJ5IHZhbHVlcyBpbiB0aGUgcmFnZ2VkIHRlbnNvci4gVGhlIGRlZmF1bHQgdmFsdWVcbiAqICAgICBtdXN0IGJlIGNvbXBhdGlibGUgd2l0aCB0aGlzIGJyb2FkY2FzdCBvcGVyYXRpb24sIGFuZCBtdXN0IGhhdmUgZmV3ZXJcbiAqICAgICBkaW1lbnNpb25zIHRoYW4gdGhlIHZhbHVlIHRlbnNvci5cbiAqIEBwYXJhbSByb3dQYXJ0aXRpb25UZW5zb3JzOiBBIGxpc3Qgb2YgYXQgbGVhc3QgMSBUZW5zb3Igb2JqZWN0cyB3aXRoIHRoZSBzYW1lXG4gKiAgICAgdHlwZSBpbjogJ2ludDMyJy5cbiAqIEBwYXJhbSByb3dQYXJ0aXRpb25UeXBlczogQSBsaXN0IG9mIHN0cmluZ3MuIFRoZSB0eXBlcyBvZiB0aGUgcm93IHBhcnRpdGlvblxuICogICAgIHRlbnNvcnMuIEF0IHByZXNlbnQsIHRoZXNlIGNhbiBiZTpcbiAqICAgICBcIlJPV19TUExJVFNcIjogdGhlIHJvd19zcGxpdHMgdGVuc29yIGZyb20gdGhlIHJhZ2dlZCB0ZW5zb3IuXG4gKiAgICAgXCJWQUxVRV9ST1dJRFNcIjogdGhlIHZhbHVlX3Jvd2lkcyB0ZW5zb3IgZnJvbSB0aGUgcmFnZ2VkIHRlbnNvci5cbiAqICAgICBcIkZJUlNUX0RJTV9TSVpFXCI6IGlmIHZhbHVlX3Jvd2lkcyBpcyB1c2VkIGZvciB0aGUgZmlyc3QgZGltZW5zaW9uLCB0aGVuXG4gKiAgICAgICAgIGl0IGlzIHByZWNlZWRlZCBieSBcIkZJUlNUX0RJTV9TSVpFXCIuIFRoZSB0ZW5zb3JzIGFyZSBpbiB0aGUgb3JkZXIgb2ZcbiAqICAgICAgICAgdGhlIGRpbWVuc2lvbnMuXG4gKiBAcmV0dXJuIEEgVGVuc29yLiBIYXMgdGhlIHNhbWUgdHlwZSBhcyB2YWx1ZXMuXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdSYWdnZWQnfVxuICovXG5mdW5jdGlvbiByYWdnZWRUZW5zb3JUb1RlbnNvcl8oXG4gICAgc2hhcGU6IFRlbnNvcnxUZW5zb3JMaWtlLCB2YWx1ZXM6IFRlbnNvcnxUZW5zb3JMaWtlLFxuICAgIGRlZmF1bHRWYWx1ZTogVGVuc29yfFRlbnNvckxpa2UsIHJvd1BhcnRpdGlvblRlbnNvcnM6IFRlbnNvcltdLFxuICAgIHJvd1BhcnRpdGlvblR5cGVzOiBzdHJpbmdbXSk6IFRlbnNvciB7XG4gIGNvbnN0ICRzaGFwZSA9XG4gICAgICBjb252ZXJ0VG9UZW5zb3Ioc2hhcGUsICdzaGFwZScsICdyYWdnZWRUZW5zb3JUb1RlbnNvcicsICdpbnQzMicpO1xuICBjb25zdCAkdmFsdWVzID0gY29udmVydFRvVGVuc29yKHZhbHVlcywgJ3ZhbHVlcycsICdyYWdnZWRUZW5zb3JUb1RlbnNvcicpO1xuICBjb25zdCAkZGVmYXVsdFZhbHVlID0gY29udmVydFRvVGVuc29yKFxuICAgICAgZGVmYXVsdFZhbHVlLCAnZGVmYXVsdFZhbHVlJywgJ3JhZ2dlZFRlbnNvclRvVGVuc29yJywgJHZhbHVlcy5kdHlwZSk7XG4gIGNvbnN0ICRyb3dQYXJ0aXRpb25UZW5zb3JzID0gcm93UGFydGl0aW9uVGVuc29ycy5tYXAoXG4gICAgICAodCwgaSkgPT5cbiAgICAgICAgICBjb252ZXJ0VG9UZW5zb3IodCwgYHRlbnNvcnMke2l9YCwgJ3JhZ2dlZFRlbnNvclRvVGVuc29yJywgJ2ludDMyJykpO1xuXG4gIGNvbnN0IGlucHV0czogUmFnZ2VkVGVuc29yVG9UZW5zb3JJbnB1dHMgPSB7XG4gICAgc2hhcGU6ICRzaGFwZSxcbiAgICB2YWx1ZXM6ICR2YWx1ZXMsXG4gICAgZGVmYXVsdFZhbHVlOiAkZGVmYXVsdFZhbHVlLFxuICAgIHJvd1BhcnRpdGlvblRlbnNvcnM6ICRyb3dQYXJ0aXRpb25UZW5zb3JzXG4gIH07XG4gIGNvbnN0IGF0dHJzOiBSYWdnZWRUZW5zb3JUb1RlbnNvckF0dHJzID0ge3Jvd1BhcnRpdGlvblR5cGVzfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChSYWdnZWRUZW5zb3JUb1RlbnNvciwgaW5wdXRzIGFzIHt9LCBhdHRycyBhcyB7fSk7XG59XG5cbmV4cG9ydCBjb25zdCByYWdnZWRUZW5zb3JUb1RlbnNvciA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7cmFnZ2VkVGVuc29yVG9UZW5zb3JffSk7XG4iXX0=