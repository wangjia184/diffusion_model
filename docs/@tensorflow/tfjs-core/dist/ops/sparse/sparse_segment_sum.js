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
import { SparseSegmentSum } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import { op } from '../operation';
/**
 * Computes the sum along sparse segments of a tensor.
 *
 * ```js
 * const c = tf.tensor2d([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]]);
 * // Select two rows, one segment.
 * const result1 = tf.sparse.sparseSegmentSum(c,
 *                                           tf.tensor1d([0, 1], 'int32'),
 *                                           tf.tensor1d([0, 0], 'int32'));
 * result1.print(); // [[0, 0, 0, 0]]
 *
 * // Select two rows, two segments.
 * const result2 = tf.sparse.sparseSegmentSum(c,
 *                                           tf.tensor1d([0, 1], 'int32'),
 *                                           tf.tensor1d([0, 1], 'int32'));
 * result2.print(); // [[1, 2, 3, 4], [-1, -2, -3, -4]]
 *
 * // Select all rows, two segments.
 * const result3 = tf.sparse.sparseSegmentSum(c,
 *                                           tf.tensor1d([0, 1, 2], 'int32'),
 *                                           tf.tensor1d([0, 0, 1], 'int32'));
 * result3.print(); // [[0, 0, 0, 0], [5, 6, 7, 8]]
 * ```
 * @param data: A Tensor of at least one dimension with data that will be
 *     assembled in the output.
 * @param indices: A 1-D Tensor with indices into data. Has same rank as
 *     segmentIds.
 * @param segmentIds: A 1-D Tensor with indices into the output Tensor. Values
 *     should be sorted and can be repeated.
 * @return Has same shape as data, except for dimension 0 which has equal to
 *         the number of segments.
 *
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
function sparseSegmentSum_(data, indices, segmentIds) {
    const $data = convertToTensor(data, 'data', 'sparseSegmentSum');
    const $indices = convertToTensor(indices, 'indices', 'sparseSegmentSum', 'int32');
    const $segmentIds = convertToTensor(segmentIds, 'segmentIds', 'sparseSegmentSum', 'int32');
    if ($data.rank < 1) {
        throw new Error(`Data should be at least 1 dimensional but received scalar`);
    }
    if ($indices.rank !== 1) {
        throw new Error(`Indices should be Tensor1D but received shape
         ${$indices.shape}`);
    }
    if ($segmentIds.rank !== 1) {
        throw new Error(`Segment ids should be Tensor1D but received shape
         ${$segmentIds.shape}`);
    }
    const inputs = {
        data: $data,
        indices: $indices,
        segmentIds: $segmentIds
    };
    return ENGINE.runKernel(SparseSegmentSum, inputs);
}
export const sparseSegmentSum = /* @__PURE__ */ op({ sparseSegmentSum_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BhcnNlX3NlZ21lbnRfc3VtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvc3BhcnNlL3NwYXJzZV9zZWdtZW50X3N1bS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ3BDLE9BQU8sRUFBQyxnQkFBZ0IsRUFBeUIsTUFBTSxvQkFBb0IsQ0FBQztBQUU1RSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUVoQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBaUNHO0FBQ0gsU0FBUyxpQkFBaUIsQ0FDdEIsSUFBdUIsRUFBRSxPQUE0QixFQUNyRCxVQUErQjtJQUNqQyxNQUFNLEtBQUssR0FBRyxlQUFlLENBQUMsSUFBSSxFQUFFLE1BQU0sRUFBRSxrQkFBa0IsQ0FBQyxDQUFDO0lBQ2hFLE1BQU0sUUFBUSxHQUNWLGVBQWUsQ0FBQyxPQUFPLEVBQUUsU0FBUyxFQUFFLGtCQUFrQixFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ3JFLE1BQU0sV0FBVyxHQUNiLGVBQWUsQ0FBQyxVQUFVLEVBQUUsWUFBWSxFQUFFLGtCQUFrQixFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBRTNFLElBQUksS0FBSyxDQUFDLElBQUksR0FBRyxDQUFDLEVBQUU7UUFDbEIsTUFBTSxJQUFJLEtBQUssQ0FDWCwyREFBMkQsQ0FBQyxDQUFDO0tBQ2xFO0lBQ0QsSUFBSSxRQUFRLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUN2QixNQUFNLElBQUksS0FBSyxDQUFDO1dBQ1QsUUFBUSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7S0FDMUI7SUFDRCxJQUFJLFdBQVcsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQzFCLE1BQU0sSUFBSSxLQUFLLENBQUM7V0FDVCxXQUFXLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUM3QjtJQUVELE1BQU0sTUFBTSxHQUEyQjtRQUNyQyxJQUFJLEVBQUUsS0FBSztRQUNYLE9BQU8sRUFBRSxRQUFRO1FBQ2pCLFVBQVUsRUFBRSxXQUFXO0tBQ3hCLENBQUM7SUFFRixPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQUMsZ0JBQWdCLEVBQUUsTUFBWSxDQUFDLENBQUM7QUFDMUQsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGdCQUFnQixHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxpQkFBaUIsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtTcGFyc2VTZWdtZW50U3VtLCBTcGFyc2VTZWdtZW50U3VtSW5wdXRzfSBmcm9tICcuLi8uLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtUZW5zb3IsIFRlbnNvcjFEfSBmcm9tICcuLi8uLi90ZW5zb3InO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uLy4uL3R5cGVzJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIHN1bSBhbG9uZyBzcGFyc2Ugc2VnbWVudHMgb2YgYSB0ZW5zb3IuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGMgPSB0Zi50ZW5zb3IyZChbWzEsMiwzLDRdLCBbLTEsLTIsLTMsLTRdLCBbNSw2LDcsOF1dKTtcbiAqIC8vIFNlbGVjdCB0d28gcm93cywgb25lIHNlZ21lbnQuXG4gKiBjb25zdCByZXN1bHQxID0gdGYuc3BhcnNlLnNwYXJzZVNlZ21lbnRTdW0oYyxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjFkKFswLCAxXSwgJ2ludDMyJyksXG4gKiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0Zi50ZW5zb3IxZChbMCwgMF0sICdpbnQzMicpKTtcbiAqIHJlc3VsdDEucHJpbnQoKTsgLy8gW1swLCAwLCAwLCAwXV1cbiAqXG4gKiAvLyBTZWxlY3QgdHdvIHJvd3MsIHR3byBzZWdtZW50cy5cbiAqIGNvbnN0IHJlc3VsdDIgPSB0Zi5zcGFyc2Uuc3BhcnNlU2VnbWVudFN1bShjLFxuICogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGYudGVuc29yMWQoWzAsIDFdLCAnaW50MzInKSxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjFkKFswLCAxXSwgJ2ludDMyJykpO1xuICogcmVzdWx0Mi5wcmludCgpOyAvLyBbWzEsIDIsIDMsIDRdLCBbLTEsIC0yLCAtMywgLTRdXVxuICpcbiAqIC8vIFNlbGVjdCBhbGwgcm93cywgdHdvIHNlZ21lbnRzLlxuICogY29uc3QgcmVzdWx0MyA9IHRmLnNwYXJzZS5zcGFyc2VTZWdtZW50U3VtKGMsXG4gKiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0Zi50ZW5zb3IxZChbMCwgMSwgMl0sICdpbnQzMicpLFxuICogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGYudGVuc29yMWQoWzAsIDAsIDFdLCAnaW50MzInKSk7XG4gKiByZXN1bHQzLnByaW50KCk7IC8vIFtbMCwgMCwgMCwgMF0sIFs1LCA2LCA3LCA4XV1cbiAqIGBgYFxuICogQHBhcmFtIGRhdGE6IEEgVGVuc29yIG9mIGF0IGxlYXN0IG9uZSBkaW1lbnNpb24gd2l0aCBkYXRhIHRoYXQgd2lsbCBiZVxuICogICAgIGFzc2VtYmxlZCBpbiB0aGUgb3V0cHV0LlxuICogQHBhcmFtIGluZGljZXM6IEEgMS1EIFRlbnNvciB3aXRoIGluZGljZXMgaW50byBkYXRhLiBIYXMgc2FtZSByYW5rIGFzXG4gKiAgICAgc2VnbWVudElkcy5cbiAqIEBwYXJhbSBzZWdtZW50SWRzOiBBIDEtRCBUZW5zb3Igd2l0aCBpbmRpY2VzIGludG8gdGhlIG91dHB1dCBUZW5zb3IuIFZhbHVlc1xuICogICAgIHNob3VsZCBiZSBzb3J0ZWQgYW5kIGNhbiBiZSByZXBlYXRlZC5cbiAqIEByZXR1cm4gSGFzIHNhbWUgc2hhcGUgYXMgZGF0YSwgZXhjZXB0IGZvciBkaW1lbnNpb24gMCB3aGljaCBoYXMgZXF1YWwgdG9cbiAqICAgICAgICAgdGhlIG51bWJlciBvZiBzZWdtZW50cy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdTcGFyc2UnfVxuICovXG5mdW5jdGlvbiBzcGFyc2VTZWdtZW50U3VtXyhcbiAgICBkYXRhOiBUZW5zb3J8VGVuc29yTGlrZSwgaW5kaWNlczogVGVuc29yMUR8VGVuc29yTGlrZSxcbiAgICBzZWdtZW50SWRzOiBUZW5zb3IxRHxUZW5zb3JMaWtlKTogVGVuc29yIHtcbiAgY29uc3QgJGRhdGEgPSBjb252ZXJ0VG9UZW5zb3IoZGF0YSwgJ2RhdGEnLCAnc3BhcnNlU2VnbWVudFN1bScpO1xuICBjb25zdCAkaW5kaWNlcyA9XG4gICAgICBjb252ZXJ0VG9UZW5zb3IoaW5kaWNlcywgJ2luZGljZXMnLCAnc3BhcnNlU2VnbWVudFN1bScsICdpbnQzMicpO1xuICBjb25zdCAkc2VnbWVudElkcyA9XG4gICAgICBjb252ZXJ0VG9UZW5zb3Ioc2VnbWVudElkcywgJ3NlZ21lbnRJZHMnLCAnc3BhcnNlU2VnbWVudFN1bScsICdpbnQzMicpO1xuXG4gIGlmICgkZGF0YS5yYW5rIDwgMSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYERhdGEgc2hvdWxkIGJlIGF0IGxlYXN0IDEgZGltZW5zaW9uYWwgYnV0IHJlY2VpdmVkIHNjYWxhcmApO1xuICB9XG4gIGlmICgkaW5kaWNlcy5yYW5rICE9PSAxKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBJbmRpY2VzIHNob3VsZCBiZSBUZW5zb3IxRCBidXQgcmVjZWl2ZWQgc2hhcGVcbiAgICAgICAgICR7JGluZGljZXMuc2hhcGV9YCk7XG4gIH1cbiAgaWYgKCRzZWdtZW50SWRzLnJhbmsgIT09IDEpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFNlZ21lbnQgaWRzIHNob3VsZCBiZSBUZW5zb3IxRCBidXQgcmVjZWl2ZWQgc2hhcGVcbiAgICAgICAgICR7JHNlZ21lbnRJZHMuc2hhcGV9YCk7XG4gIH1cblxuICBjb25zdCBpbnB1dHM6IFNwYXJzZVNlZ21lbnRTdW1JbnB1dHMgPSB7XG4gICAgZGF0YTogJGRhdGEsXG4gICAgaW5kaWNlczogJGluZGljZXMsXG4gICAgc2VnbWVudElkczogJHNlZ21lbnRJZHNcbiAgfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChTcGFyc2VTZWdtZW50U3VtLCBpbnB1dHMgYXMge30pO1xufVxuXG5leHBvcnQgY29uc3Qgc3BhcnNlU2VnbWVudFN1bSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7c3BhcnNlU2VnbWVudFN1bV99KTtcbiJdfQ==