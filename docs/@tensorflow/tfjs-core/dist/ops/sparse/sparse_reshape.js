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
import { SparseReshape } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import { op } from '../operation';
/**
 * This operation has the same semantics as reshape on the represented dense
 * tensor. The `inputIndices` are recomputed based on the requested `newShape`.
 * If one component of `newShape` is the special value -1, the size of that
 * dimension is computed so that the total dense size remains constant. At most
 * one component of `newShape` can be -1. The number of dense elements implied
 * by `newShape` must be the same as the number of dense elements originally
 * implied by `inputShape`. Reshaping does not affect the order of values in the
 * SparseTensor. If the input tensor has rank R_in and N non-empty values, and
 * `newShape` has length R_out, then `inputIndices` has shape [N, R_in],
 * `inputShape` has length R_in, `outputIndices` has shape [N, R_out], and
 * `outputShape` has length R_out.
 *
 * ```js
 * const result = tf.sparse.sparseReshape(
 *   [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 2, 3]],
 *   [2, 3, 6], [9, -1]);
 * console.log(result);
 * result['outputIndices'].print(); //[[0, 0], [0, 1], [1, 2], [4, 2], [8, 1]]
 * result['outputShape'].print(); // [9, 4]
 * ```
 * @param inputIndices: 2-D. N x R_in matrix with the indices of non-empty
 * values in a SparseTensor.
 * @param inputShape: 1-D. R_in Tensor1D with the input SparseTensor's dense
 * shape.
 * @param newShape: 1-D. R_out Tensor1D with the requested new dense shape.
 * @return A map with the following properties:
 *     - outputIndices: 2-D. N x R_out matrix with the updated indices of
 *       non-empty values in the output SparseTensor.
 *     - outputShape: 1-D. R_out vector with the full dense shape of the output
 *       SparseTensor. This is the same as newShape but with any -1 dimensions
 *        filled in.
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
function sparseReshape_(inputIndices, inputShape, newShape) {
    const $inputIndices = convertToTensor(inputIndices, 'inputIndices', 'sparseReshape', 'int32');
    const $inputShape = convertToTensor(inputShape, 'inputShape', 'sparseReshape', 'int32');
    const $newShape = convertToTensor(newShape, 'newShape', 'sparseReshape', 'int32');
    if ($inputIndices.rank !== 2) {
        throw new Error(`Input indices should be Tensor2D but received shape
        ${$inputIndices.shape}`);
    }
    if ($inputShape.rank !== 1) {
        throw new Error(`Input shape should be Tensor1D but received shape ${$inputShape.shape}`);
    }
    if ($newShape.rank !== 1) {
        throw new Error(`New shape should be Tensor1D but received shape ${$newShape.shape}`);
    }
    const inputs = {
        inputIndices: $inputIndices,
        inputShape: $inputShape,
        newShape: $newShape
    };
    const result = ENGINE.runKernel(SparseReshape, inputs);
    return { outputIndices: result[0], outputShape: result[1] };
}
export const sparseReshape = /* @__PURE__ */ op({ sparseReshape_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BhcnNlX3Jlc2hhcGUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9zcGFyc2Uvc3BhcnNlX3Jlc2hhcGUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNwQyxPQUFPLEVBQUMsYUFBYSxFQUFzQixNQUFNLG9CQUFvQixDQUFDO0FBR3RFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV0RCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRWhDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FpQ0c7QUFDSCxTQUFTLGNBQWMsQ0FDbkIsWUFBaUMsRUFBRSxVQUErQixFQUNsRSxRQUE2QjtJQUMvQixNQUFNLGFBQWEsR0FDZixlQUFlLENBQUMsWUFBWSxFQUFFLGNBQWMsRUFBRSxlQUFlLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDNUUsTUFBTSxXQUFXLEdBQ2IsZUFBZSxDQUFDLFVBQVUsRUFBRSxZQUFZLEVBQUUsZUFBZSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ3hFLE1BQU0sU0FBUyxHQUNYLGVBQWUsQ0FBQyxRQUFRLEVBQUUsVUFBVSxFQUFFLGVBQWUsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUVwRSxJQUFJLGFBQWEsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQzVCLE1BQU0sSUFBSSxLQUFLLENBQUM7VUFDVixhQUFhLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUM5QjtJQUNELElBQUksV0FBVyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDMUIsTUFBTSxJQUFJLEtBQUssQ0FBQyxxREFDWixXQUFXLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUMxQjtJQUNELElBQUksU0FBUyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDeEIsTUFBTSxJQUFJLEtBQUssQ0FDWCxtREFBbUQsU0FBUyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7S0FDM0U7SUFFRCxNQUFNLE1BQU0sR0FBd0I7UUFDbEMsWUFBWSxFQUFFLGFBQWE7UUFDM0IsVUFBVSxFQUFFLFdBQVc7UUFDdkIsUUFBUSxFQUFFLFNBQVM7S0FDcEIsQ0FBQztJQUNGLE1BQU0sTUFBTSxHQUFhLE1BQU0sQ0FBQyxTQUFTLENBQUMsYUFBYSxFQUFFLE1BQVksQ0FBQyxDQUFDO0lBQ3ZFLE9BQU8sRUFBQyxhQUFhLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUMsQ0FBQztBQUM1RCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sYUFBYSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxjQUFjLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vLi4vZW5naW5lJztcbmltcG9ydCB7U3BhcnNlUmVzaGFwZSwgU3BhcnNlUmVzaGFwZUlucHV0c30gZnJvbSAnLi4vLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7VGVuc29yLCBUZW5zb3IxRCwgVGVuc29yMkR9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi8uLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uLy4uL3R5cGVzJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5cbi8qKlxuICogVGhpcyBvcGVyYXRpb24gaGFzIHRoZSBzYW1lIHNlbWFudGljcyBhcyByZXNoYXBlIG9uIHRoZSByZXByZXNlbnRlZCBkZW5zZVxuICogdGVuc29yLiBUaGUgYGlucHV0SW5kaWNlc2AgYXJlIHJlY29tcHV0ZWQgYmFzZWQgb24gdGhlIHJlcXVlc3RlZCBgbmV3U2hhcGVgLlxuICogSWYgb25lIGNvbXBvbmVudCBvZiBgbmV3U2hhcGVgIGlzIHRoZSBzcGVjaWFsIHZhbHVlIC0xLCB0aGUgc2l6ZSBvZiB0aGF0XG4gKiBkaW1lbnNpb24gaXMgY29tcHV0ZWQgc28gdGhhdCB0aGUgdG90YWwgZGVuc2Ugc2l6ZSByZW1haW5zIGNvbnN0YW50LiBBdCBtb3N0XG4gKiBvbmUgY29tcG9uZW50IG9mIGBuZXdTaGFwZWAgY2FuIGJlIC0xLiBUaGUgbnVtYmVyIG9mIGRlbnNlIGVsZW1lbnRzIGltcGxpZWRcbiAqIGJ5IGBuZXdTaGFwZWAgbXVzdCBiZSB0aGUgc2FtZSBhcyB0aGUgbnVtYmVyIG9mIGRlbnNlIGVsZW1lbnRzIG9yaWdpbmFsbHlcbiAqIGltcGxpZWQgYnkgYGlucHV0U2hhcGVgLiBSZXNoYXBpbmcgZG9lcyBub3QgYWZmZWN0IHRoZSBvcmRlciBvZiB2YWx1ZXMgaW4gdGhlXG4gKiBTcGFyc2VUZW5zb3IuIElmIHRoZSBpbnB1dCB0ZW5zb3IgaGFzIHJhbmsgUl9pbiBhbmQgTiBub24tZW1wdHkgdmFsdWVzLCBhbmRcbiAqIGBuZXdTaGFwZWAgaGFzIGxlbmd0aCBSX291dCwgdGhlbiBgaW5wdXRJbmRpY2VzYCBoYXMgc2hhcGUgW04sIFJfaW5dLFxuICogYGlucHV0U2hhcGVgIGhhcyBsZW5ndGggUl9pbiwgYG91dHB1dEluZGljZXNgIGhhcyBzaGFwZSBbTiwgUl9vdXRdLCBhbmRcbiAqIGBvdXRwdXRTaGFwZWAgaGFzIGxlbmd0aCBSX291dC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgcmVzdWx0ID0gdGYuc3BhcnNlLnNwYXJzZVJlc2hhcGUoXG4gKiAgIFtbMCwgMCwgMF0sIFswLCAwLCAxXSwgWzAsIDEsIDBdLCBbMSwgMCwgMF0sIFsxLCAyLCAzXV0sXG4gKiAgIFsyLCAzLCA2XSwgWzksIC0xXSk7XG4gKiBjb25zb2xlLmxvZyhyZXN1bHQpO1xuICogcmVzdWx0WydvdXRwdXRJbmRpY2VzJ10ucHJpbnQoKTsgLy9bWzAsIDBdLCBbMCwgMV0sIFsxLCAyXSwgWzQsIDJdLCBbOCwgMV1dXG4gKiByZXN1bHRbJ291dHB1dFNoYXBlJ10ucHJpbnQoKTsgLy8gWzksIDRdXG4gKiBgYGBcbiAqIEBwYXJhbSBpbnB1dEluZGljZXM6IDItRC4gTiB4IFJfaW4gbWF0cml4IHdpdGggdGhlIGluZGljZXMgb2Ygbm9uLWVtcHR5XG4gKiB2YWx1ZXMgaW4gYSBTcGFyc2VUZW5zb3IuXG4gKiBAcGFyYW0gaW5wdXRTaGFwZTogMS1ELiBSX2luIFRlbnNvcjFEIHdpdGggdGhlIGlucHV0IFNwYXJzZVRlbnNvcidzIGRlbnNlXG4gKiBzaGFwZS5cbiAqIEBwYXJhbSBuZXdTaGFwZTogMS1ELiBSX291dCBUZW5zb3IxRCB3aXRoIHRoZSByZXF1ZXN0ZWQgbmV3IGRlbnNlIHNoYXBlLlxuICogQHJldHVybiBBIG1hcCB3aXRoIHRoZSBmb2xsb3dpbmcgcHJvcGVydGllczpcbiAqICAgICAtIG91dHB1dEluZGljZXM6IDItRC4gTiB4IFJfb3V0IG1hdHJpeCB3aXRoIHRoZSB1cGRhdGVkIGluZGljZXMgb2ZcbiAqICAgICAgIG5vbi1lbXB0eSB2YWx1ZXMgaW4gdGhlIG91dHB1dCBTcGFyc2VUZW5zb3IuXG4gKiAgICAgLSBvdXRwdXRTaGFwZTogMS1ELiBSX291dCB2ZWN0b3Igd2l0aCB0aGUgZnVsbCBkZW5zZSBzaGFwZSBvZiB0aGUgb3V0cHV0XG4gKiAgICAgICBTcGFyc2VUZW5zb3IuIFRoaXMgaXMgdGhlIHNhbWUgYXMgbmV3U2hhcGUgYnV0IHdpdGggYW55IC0xIGRpbWVuc2lvbnNcbiAqICAgICAgICBmaWxsZWQgaW4uXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdTcGFyc2UnfVxuICovXG5mdW5jdGlvbiBzcGFyc2VSZXNoYXBlXyhcbiAgICBpbnB1dEluZGljZXM6IFRlbnNvcjJEfFRlbnNvckxpa2UsIGlucHV0U2hhcGU6IFRlbnNvcjFEfFRlbnNvckxpa2UsXG4gICAgbmV3U2hhcGU6IFRlbnNvcjFEfFRlbnNvckxpa2UpOiBOYW1lZFRlbnNvck1hcCB7XG4gIGNvbnN0ICRpbnB1dEluZGljZXMgPVxuICAgICAgY29udmVydFRvVGVuc29yKGlucHV0SW5kaWNlcywgJ2lucHV0SW5kaWNlcycsICdzcGFyc2VSZXNoYXBlJywgJ2ludDMyJyk7XG4gIGNvbnN0ICRpbnB1dFNoYXBlID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihpbnB1dFNoYXBlLCAnaW5wdXRTaGFwZScsICdzcGFyc2VSZXNoYXBlJywgJ2ludDMyJyk7XG4gIGNvbnN0ICRuZXdTaGFwZSA9XG4gICAgICBjb252ZXJ0VG9UZW5zb3IobmV3U2hhcGUsICduZXdTaGFwZScsICdzcGFyc2VSZXNoYXBlJywgJ2ludDMyJyk7XG5cbiAgaWYgKCRpbnB1dEluZGljZXMucmFuayAhPT0gMikge1xuICAgIHRocm93IG5ldyBFcnJvcihgSW5wdXQgaW5kaWNlcyBzaG91bGQgYmUgVGVuc29yMkQgYnV0IHJlY2VpdmVkIHNoYXBlXG4gICAgICAgICR7JGlucHV0SW5kaWNlcy5zaGFwZX1gKTtcbiAgfVxuICBpZiAoJGlucHV0U2hhcGUucmFuayAhPT0gMSkge1xuICAgIHRocm93IG5ldyBFcnJvcihgSW5wdXQgc2hhcGUgc2hvdWxkIGJlIFRlbnNvcjFEIGJ1dCByZWNlaXZlZCBzaGFwZSAke1xuICAgICAgICAkaW5wdXRTaGFwZS5zaGFwZX1gKTtcbiAgfVxuICBpZiAoJG5ld1NoYXBlLnJhbmsgIT09IDEpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBOZXcgc2hhcGUgc2hvdWxkIGJlIFRlbnNvcjFEIGJ1dCByZWNlaXZlZCBzaGFwZSAkeyRuZXdTaGFwZS5zaGFwZX1gKTtcbiAgfVxuXG4gIGNvbnN0IGlucHV0czogU3BhcnNlUmVzaGFwZUlucHV0cyA9IHtcbiAgICBpbnB1dEluZGljZXM6ICRpbnB1dEluZGljZXMsXG4gICAgaW5wdXRTaGFwZTogJGlucHV0U2hhcGUsXG4gICAgbmV3U2hhcGU6ICRuZXdTaGFwZVxuICB9O1xuICBjb25zdCByZXN1bHQ6IFRlbnNvcltdID0gRU5HSU5FLnJ1bktlcm5lbChTcGFyc2VSZXNoYXBlLCBpbnB1dHMgYXMge30pO1xuICByZXR1cm4ge291dHB1dEluZGljZXM6IHJlc3VsdFswXSwgb3V0cHV0U2hhcGU6IHJlc3VsdFsxXX07XG59XG5cbmV4cG9ydCBjb25zdCBzcGFyc2VSZXNoYXBlID0gLyogQF9fUFVSRV9fICovIG9wKHtzcGFyc2VSZXNoYXBlX30pO1xuIl19