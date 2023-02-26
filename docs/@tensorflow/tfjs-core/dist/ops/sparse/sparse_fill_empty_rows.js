/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import { ENGINE } from '../../engine';
import { SparseFillEmptyRows } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import { op } from '../operation';
/**
 * The input SparseTensor is represented via the map of inputs {`indices`,
 * `values`, `denseShape`}. The output SparseTensor has the same `denseShape`
 * but with indices `outputIndices` and values `outputValues`. This op inserts a
 * single entry for every row that doesn't have any values. The index is created
 * as `[row, 0, ..., 0]` and the inserted value is `defaultValue`.
 *
 * For example, suppose `spInput` has shape [5, 6] and non-empty values:
 * [0, 1]: a
 * [0, 3]: b
 * [2, 0]: c
 * [3, 1]: d
 *
 * Rows 1 and 4 are empty, so the output will be of shape [5, 6] with values:
 * [0, 1]: a
 * [0, 3]: b
 * [1, 0]: `defaultValue`
 * [2, 0]: c
 * [3, 1]: d
 * [4, 0]: `defaultValue`
 *
 * The output SparseTensor will be in row-major order and will have the same
 * shape as the input.
 *
 * This op also returns an indicator vector shaped [dense_shape[0]] such that
 * emptyRowIndicator[i] = True iff row i was an empty row.
 *
 * And a reverse index map vector shaped [indices.shape[0]] that is used during
 * backpropagation, reverseIndexMap[i] = outi s.t. indices[i, j] ==
 * outputIndices[outi, j] for all j
 *
 * ```js
 * const result = tf.sparse.sparseFillEmptyRows(
 *   [[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]],
 *   [0, 10, 13, 14, 32, 33], [5, 6], -1);
 * console.log(result);
 * result['outputIndices'].print(); // [[0, 0], [1, 0], [1, 3], [1, 4],
 *                                  //  [2, 0], [3, 2], [3, 3], [4, 0]]
 * result['outputValues'].print(); // [0, 10, 13, 14,-1, 32, 33, -1]
 * result['emptyRowIndicator'].print(); // [false, false, true, false, true]
 * result['reverseIndexMap'].print(); // [0, 1, 2, 3, 5, 6]
 * ```
 * @param indices: 2-D. The indices of the sparse tensor.
 * @param values: 1-D. The values of the sparse tensor.
 * @param denseShape: 1-D. The shape of the sparse tensor.
 * @param defaultValue: 0-D. Default value to insert into location [row, 0, ...,
 *     0] for rows missing from the input sparse tensor.
 * @return A map with the following properties:
 *     - outputIndices
 *     - outputValues: 1-D. The values of the filled sparse tensor.
 *     - emptyRowIndicator: 1-D. Whether the dense row was missing in the input
 * sparse tensor.
 *     - reverseIndexMap: 1-D. A map from the input indices to the output
 * indices.
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
function sparseFillEmptyRows_(indices, values, denseShape, defaultValue) {
    const $indices = convertToTensor(indices, 'indices', 'sparseFillEmptyRows', 'int32');
    const $values = convertToTensor(values, 'values', 'sparseFillEmptyRows');
    const $denseShape = convertToTensor(denseShape, 'denseShape', 'sparseFillEmptyRows', 'int32');
    const $defaultValue = convertToTensor(defaultValue, 'defaultValue', 'sparseFillEmptyRows', $values.dtype);
    if ($indices.rank !== 2) {
        throw new Error(`Indices should be Tensor2D but received shape
        ${$indices.shape}`);
    }
    if ($values.rank !== 1) {
        throw new Error(`Values should be Tensor1D but received shape ${$values.shape}`);
    }
    if ($denseShape.rank !== 1) {
        throw new Error(`Dense shape should be Tensor1D but received shape ${$denseShape.shape}`);
    }
    if ($defaultValue.rank !== 0) {
        throw new Error(`Default value should be a scalar but received shape ${$defaultValue.shape}`);
    }
    const inputs = {
        indices: $indices,
        values: $values,
        denseShape: $denseShape,
        defaultValue: $defaultValue
    };
    const result = ENGINE.runKernel(SparseFillEmptyRows, inputs);
    return {
        outputIndices: result[0],
        outputValues: result[1],
        emptyRowIndicator: result[2],
        reverseIndexMap: result[3]
    };
}
export const sparseFillEmptyRows = /* @__PURE__ */ op({ sparseFillEmptyRows_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BhcnNlX2ZpbGxfZW1wdHlfcm93cy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NwYXJzZS9zcGFyc2VfZmlsbF9lbXB0eV9yb3dzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLG1CQUFtQixFQUE0QixNQUFNLG9CQUFvQixDQUFDO0FBR2xGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV0RCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRWhDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBdURHO0FBQ0gsU0FBUyxvQkFBb0IsQ0FDekIsT0FBNEIsRUFBRSxNQUEyQixFQUN6RCxVQUErQixFQUMvQixZQUErQjtJQUNqQyxNQUFNLFFBQVEsR0FDVixlQUFlLENBQUMsT0FBTyxFQUFFLFNBQVMsRUFBRSxxQkFBcUIsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUN4RSxNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxxQkFBcUIsQ0FBQyxDQUFDO0lBQ3pFLE1BQU0sV0FBVyxHQUNiLGVBQWUsQ0FBQyxVQUFVLEVBQUUsWUFBWSxFQUFFLHFCQUFxQixFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQzlFLE1BQU0sYUFBYSxHQUFHLGVBQWUsQ0FDakMsWUFBWSxFQUFFLGNBQWMsRUFBRSxxQkFBcUIsRUFBRSxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUM7SUFFeEUsSUFBSSxRQUFRLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUN2QixNQUFNLElBQUksS0FBSyxDQUFDO1VBQ1YsUUFBUSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7S0FDekI7SUFDRCxJQUFJLE9BQU8sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ3RCLE1BQU0sSUFBSSxLQUFLLENBQ1gsZ0RBQWdELE9BQU8sQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQ3RFO0lBQ0QsSUFBSSxXQUFXLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUMxQixNQUFNLElBQUksS0FBSyxDQUFDLHFEQUNaLFdBQVcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzFCO0lBQ0QsSUFBSSxhQUFhLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUM1QixNQUFNLElBQUksS0FBSyxDQUFDLHVEQUNaLGFBQWEsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzVCO0lBRUQsTUFBTSxNQUFNLEdBQThCO1FBQ3hDLE9BQU8sRUFBRSxRQUFRO1FBQ2pCLE1BQU0sRUFBRSxPQUFPO1FBQ2YsVUFBVSxFQUFFLFdBQVc7UUFDdkIsWUFBWSxFQUFFLGFBQWE7S0FDNUIsQ0FBQztJQUVGLE1BQU0sTUFBTSxHQUFhLE1BQU0sQ0FBQyxTQUFTLENBQUMsbUJBQW1CLEVBQUUsTUFBWSxDQUFDLENBQUM7SUFDN0UsT0FBTztRQUNMLGFBQWEsRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLFlBQVksRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDO1FBQ3ZCLGlCQUFpQixFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUM7UUFDNUIsZUFBZSxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUM7S0FDM0IsQ0FBQztBQUNKLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxtQkFBbUIsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsb0JBQW9CLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vLi4vZW5naW5lJztcbmltcG9ydCB7U3BhcnNlRmlsbEVtcHR5Um93cywgU3BhcnNlRmlsbEVtcHR5Um93c0lucHV0c30gZnJvbSAnLi4vLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7U2NhbGFyLCBUZW5zb3IsIFRlbnNvcjFELCBUZW5zb3IyRH0gZnJvbSAnLi4vLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uLy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7U2NhbGFyTGlrZSwgVGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBUaGUgaW5wdXQgU3BhcnNlVGVuc29yIGlzIHJlcHJlc2VudGVkIHZpYSB0aGUgbWFwIG9mIGlucHV0cyB7YGluZGljZXNgLFxuICogYHZhbHVlc2AsIGBkZW5zZVNoYXBlYH0uIFRoZSBvdXRwdXQgU3BhcnNlVGVuc29yIGhhcyB0aGUgc2FtZSBgZGVuc2VTaGFwZWBcbiAqIGJ1dCB3aXRoIGluZGljZXMgYG91dHB1dEluZGljZXNgIGFuZCB2YWx1ZXMgYG91dHB1dFZhbHVlc2AuIFRoaXMgb3AgaW5zZXJ0cyBhXG4gKiBzaW5nbGUgZW50cnkgZm9yIGV2ZXJ5IHJvdyB0aGF0IGRvZXNuJ3QgaGF2ZSBhbnkgdmFsdWVzLiBUaGUgaW5kZXggaXMgY3JlYXRlZFxuICogYXMgYFtyb3csIDAsIC4uLiwgMF1gIGFuZCB0aGUgaW5zZXJ0ZWQgdmFsdWUgaXMgYGRlZmF1bHRWYWx1ZWAuXG4gKlxuICogRm9yIGV4YW1wbGUsIHN1cHBvc2UgYHNwSW5wdXRgIGhhcyBzaGFwZSBbNSwgNl0gYW5kIG5vbi1lbXB0eSB2YWx1ZXM6XG4gKiBbMCwgMV06IGFcbiAqIFswLCAzXTogYlxuICogWzIsIDBdOiBjXG4gKiBbMywgMV06IGRcbiAqXG4gKiBSb3dzIDEgYW5kIDQgYXJlIGVtcHR5LCBzbyB0aGUgb3V0cHV0IHdpbGwgYmUgb2Ygc2hhcGUgWzUsIDZdIHdpdGggdmFsdWVzOlxuICogWzAsIDFdOiBhXG4gKiBbMCwgM106IGJcbiAqIFsxLCAwXTogYGRlZmF1bHRWYWx1ZWBcbiAqIFsyLCAwXTogY1xuICogWzMsIDFdOiBkXG4gKiBbNCwgMF06IGBkZWZhdWx0VmFsdWVgXG4gKlxuICogVGhlIG91dHB1dCBTcGFyc2VUZW5zb3Igd2lsbCBiZSBpbiByb3ctbWFqb3Igb3JkZXIgYW5kIHdpbGwgaGF2ZSB0aGUgc2FtZVxuICogc2hhcGUgYXMgdGhlIGlucHV0LlxuICpcbiAqIFRoaXMgb3AgYWxzbyByZXR1cm5zIGFuIGluZGljYXRvciB2ZWN0b3Igc2hhcGVkIFtkZW5zZV9zaGFwZVswXV0gc3VjaCB0aGF0XG4gKiBlbXB0eVJvd0luZGljYXRvcltpXSA9IFRydWUgaWZmIHJvdyBpIHdhcyBhbiBlbXB0eSByb3cuXG4gKlxuICogQW5kIGEgcmV2ZXJzZSBpbmRleCBtYXAgdmVjdG9yIHNoYXBlZCBbaW5kaWNlcy5zaGFwZVswXV0gdGhhdCBpcyB1c2VkIGR1cmluZ1xuICogYmFja3Byb3BhZ2F0aW9uLCByZXZlcnNlSW5kZXhNYXBbaV0gPSBvdXRpIHMudC4gaW5kaWNlc1tpLCBqXSA9PVxuICogb3V0cHV0SW5kaWNlc1tvdXRpLCBqXSBmb3IgYWxsIGpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgcmVzdWx0ID0gdGYuc3BhcnNlLnNwYXJzZUZpbGxFbXB0eVJvd3MoXG4gKiAgIFtbMCwgMF0sIFsxLCAwXSwgWzEsIDNdLCBbMSwgNF0sIFszLCAyXSwgWzMsIDNdXSxcbiAqICAgWzAsIDEwLCAxMywgMTQsIDMyLCAzM10sIFs1LCA2XSwgLTEpO1xuICogY29uc29sZS5sb2cocmVzdWx0KTtcbiAqIHJlc3VsdFsnb3V0cHV0SW5kaWNlcyddLnByaW50KCk7IC8vIFtbMCwgMF0sIFsxLCAwXSwgWzEsIDNdLCBbMSwgNF0sXG4gKiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyAgWzIsIDBdLCBbMywgMl0sIFszLCAzXSwgWzQsIDBdXVxuICogcmVzdWx0WydvdXRwdXRWYWx1ZXMnXS5wcmludCgpOyAvLyBbMCwgMTAsIDEzLCAxNCwtMSwgMzIsIDMzLCAtMV1cbiAqIHJlc3VsdFsnZW1wdHlSb3dJbmRpY2F0b3InXS5wcmludCgpOyAvLyBbZmFsc2UsIGZhbHNlLCB0cnVlLCBmYWxzZSwgdHJ1ZV1cbiAqIHJlc3VsdFsncmV2ZXJzZUluZGV4TWFwJ10ucHJpbnQoKTsgLy8gWzAsIDEsIDIsIDMsIDUsIDZdXG4gKiBgYGBcbiAqIEBwYXJhbSBpbmRpY2VzOiAyLUQuIFRoZSBpbmRpY2VzIG9mIHRoZSBzcGFyc2UgdGVuc29yLlxuICogQHBhcmFtIHZhbHVlczogMS1ELiBUaGUgdmFsdWVzIG9mIHRoZSBzcGFyc2UgdGVuc29yLlxuICogQHBhcmFtIGRlbnNlU2hhcGU6IDEtRC4gVGhlIHNoYXBlIG9mIHRoZSBzcGFyc2UgdGVuc29yLlxuICogQHBhcmFtIGRlZmF1bHRWYWx1ZTogMC1ELiBEZWZhdWx0IHZhbHVlIHRvIGluc2VydCBpbnRvIGxvY2F0aW9uIFtyb3csIDAsIC4uLixcbiAqICAgICAwXSBmb3Igcm93cyBtaXNzaW5nIGZyb20gdGhlIGlucHV0IHNwYXJzZSB0ZW5zb3IuXG4gKiBAcmV0dXJuIEEgbWFwIHdpdGggdGhlIGZvbGxvd2luZyBwcm9wZXJ0aWVzOlxuICogICAgIC0gb3V0cHV0SW5kaWNlc1xuICogICAgIC0gb3V0cHV0VmFsdWVzOiAxLUQuIFRoZSB2YWx1ZXMgb2YgdGhlIGZpbGxlZCBzcGFyc2UgdGVuc29yLlxuICogICAgIC0gZW1wdHlSb3dJbmRpY2F0b3I6IDEtRC4gV2hldGhlciB0aGUgZGVuc2Ugcm93IHdhcyBtaXNzaW5nIGluIHRoZSBpbnB1dFxuICogc3BhcnNlIHRlbnNvci5cbiAqICAgICAtIHJldmVyc2VJbmRleE1hcDogMS1ELiBBIG1hcCBmcm9tIHRoZSBpbnB1dCBpbmRpY2VzIHRvIHRoZSBvdXRwdXRcbiAqIGluZGljZXMuXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdTcGFyc2UnfVxuICovXG5mdW5jdGlvbiBzcGFyc2VGaWxsRW1wdHlSb3dzXyhcbiAgICBpbmRpY2VzOiBUZW5zb3IyRHxUZW5zb3JMaWtlLCB2YWx1ZXM6IFRlbnNvcjFEfFRlbnNvckxpa2UsXG4gICAgZGVuc2VTaGFwZTogVGVuc29yMUR8VGVuc29yTGlrZSxcbiAgICBkZWZhdWx0VmFsdWU6IFNjYWxhcnxTY2FsYXJMaWtlKTogTmFtZWRUZW5zb3JNYXAge1xuICBjb25zdCAkaW5kaWNlcyA9XG4gICAgICBjb252ZXJ0VG9UZW5zb3IoaW5kaWNlcywgJ2luZGljZXMnLCAnc3BhcnNlRmlsbEVtcHR5Um93cycsICdpbnQzMicpO1xuICBjb25zdCAkdmFsdWVzID0gY29udmVydFRvVGVuc29yKHZhbHVlcywgJ3ZhbHVlcycsICdzcGFyc2VGaWxsRW1wdHlSb3dzJyk7XG4gIGNvbnN0ICRkZW5zZVNoYXBlID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihkZW5zZVNoYXBlLCAnZGVuc2VTaGFwZScsICdzcGFyc2VGaWxsRW1wdHlSb3dzJywgJ2ludDMyJyk7XG4gIGNvbnN0ICRkZWZhdWx0VmFsdWUgPSBjb252ZXJ0VG9UZW5zb3IoXG4gICAgICBkZWZhdWx0VmFsdWUsICdkZWZhdWx0VmFsdWUnLCAnc3BhcnNlRmlsbEVtcHR5Um93cycsICR2YWx1ZXMuZHR5cGUpO1xuXG4gIGlmICgkaW5kaWNlcy5yYW5rICE9PSAyKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBJbmRpY2VzIHNob3VsZCBiZSBUZW5zb3IyRCBidXQgcmVjZWl2ZWQgc2hhcGVcbiAgICAgICAgJHskaW5kaWNlcy5zaGFwZX1gKTtcbiAgfVxuICBpZiAoJHZhbHVlcy5yYW5rICE9PSAxKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgVmFsdWVzIHNob3VsZCBiZSBUZW5zb3IxRCBidXQgcmVjZWl2ZWQgc2hhcGUgJHskdmFsdWVzLnNoYXBlfWApO1xuICB9XG4gIGlmICgkZGVuc2VTaGFwZS5yYW5rICE9PSAxKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBEZW5zZSBzaGFwZSBzaG91bGQgYmUgVGVuc29yMUQgYnV0IHJlY2VpdmVkIHNoYXBlICR7XG4gICAgICAgICRkZW5zZVNoYXBlLnNoYXBlfWApO1xuICB9XG4gIGlmICgkZGVmYXVsdFZhbHVlLnJhbmsgIT09IDApIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYERlZmF1bHQgdmFsdWUgc2hvdWxkIGJlIGEgc2NhbGFyIGJ1dCByZWNlaXZlZCBzaGFwZSAke1xuICAgICAgICAkZGVmYXVsdFZhbHVlLnNoYXBlfWApO1xuICB9XG5cbiAgY29uc3QgaW5wdXRzOiBTcGFyc2VGaWxsRW1wdHlSb3dzSW5wdXRzID0ge1xuICAgIGluZGljZXM6ICRpbmRpY2VzLFxuICAgIHZhbHVlczogJHZhbHVlcyxcbiAgICBkZW5zZVNoYXBlOiAkZGVuc2VTaGFwZSxcbiAgICBkZWZhdWx0VmFsdWU6ICRkZWZhdWx0VmFsdWVcbiAgfTtcblxuICBjb25zdCByZXN1bHQ6IFRlbnNvcltdID0gRU5HSU5FLnJ1bktlcm5lbChTcGFyc2VGaWxsRW1wdHlSb3dzLCBpbnB1dHMgYXMge30pO1xuICByZXR1cm4ge1xuICAgIG91dHB1dEluZGljZXM6IHJlc3VsdFswXSxcbiAgICBvdXRwdXRWYWx1ZXM6IHJlc3VsdFsxXSxcbiAgICBlbXB0eVJvd0luZGljYXRvcjogcmVzdWx0WzJdLFxuICAgIHJldmVyc2VJbmRleE1hcDogcmVzdWx0WzNdXG4gIH07XG59XG5cbmV4cG9ydCBjb25zdCBzcGFyc2VGaWxsRW1wdHlSb3dzID0gLyogQF9fUFVSRV9fICovIG9wKHtzcGFyc2VGaWxsRW1wdHlSb3dzX30pO1xuIl19