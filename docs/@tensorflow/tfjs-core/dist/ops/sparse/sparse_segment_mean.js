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
import { SparseSegmentMean } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import { op } from '../operation';
/**
 * Computes the mean along sparse segments of a tensor.
 *
 * ```js
 * const c = tf.tensor2d([[1,2,3,4], [-1,-2,-3,-4], [6,7,8,9]]);
 * // Select two rows, one segment.
 * const result1 = tf.sparse.sparseSegmentMean(c,
 *                                           tf.tensor1d([0, 1], 'int32'),
 *                                           tf.tensor1d([0, 0], 'int32'));
 * result1.print(); // [[0, 0, 0, 0]]
 *
 * // Select two rows, two segments.
 * const result2 = tf.sparse.sparseSegmentMean(c,
 *                                             tf.tensor1d([0, 1], 'int32'),
 *                                             tf.tensor1d([0, 1], 'int32'));
 * result2.print(); // [[1, 2, 3, 4], [-1, -2, -3, -4]]
 *
 * // Select all rows, two segments.
 * const result3 = tf.sparse.sparseSegmentMean(c,
 *                                             tf.tensor1d([0, 1, 2], 'int32'),
 *                                             tf.tensor1d([0, 1, 1], 'int32'));
 * result3.print(); // [[1.0, 2.0, 3.0, 4.0], [2.5, 2.5, 2.5, 2.5]]
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
function sparseSegmentMean_(data, indices, segmentIds) {
    const $data = convertToTensor(data, 'data', 'sparseSegmentMean');
    const $indices = convertToTensor(indices, 'indices', 'sparseSegmentMean', 'int32');
    const $segmentIds = convertToTensor(segmentIds, 'segmentIds', 'sparseSegmentMean', 'int32');
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
    return ENGINE.runKernel(SparseSegmentMean, inputs);
}
export const sparseSegmentMean = /* @__PURE__ */ op({ sparseSegmentMean_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BhcnNlX3NlZ21lbnRfbWVhbi5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NwYXJzZS9zcGFyc2Vfc2VnbWVudF9tZWFuLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLGlCQUFpQixFQUEwQixNQUFNLG9CQUFvQixDQUFDO0FBRTlFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV0RCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRWhDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FpQ0c7QUFDSCxTQUFTLGtCQUFrQixDQUN2QixJQUF1QixFQUFFLE9BQTRCLEVBQ3JELFVBQStCO0lBQ2pDLE1BQU0sS0FBSyxHQUFHLGVBQWUsQ0FBQyxJQUFJLEVBQUUsTUFBTSxFQUFFLG1CQUFtQixDQUFDLENBQUM7SUFDakUsTUFBTSxRQUFRLEdBQ1YsZUFBZSxDQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUsbUJBQW1CLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDdEUsTUFBTSxXQUFXLEdBQ2IsZUFBZSxDQUFDLFVBQVUsRUFBRSxZQUFZLEVBQUUsbUJBQW1CLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFFNUUsSUFBSSxLQUFLLENBQUMsSUFBSSxHQUFHLENBQUMsRUFBRTtRQUNsQixNQUFNLElBQUksS0FBSyxDQUNYLDJEQUEyRCxDQUFDLENBQUM7S0FDbEU7SUFDRCxJQUFJLFFBQVEsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ3ZCLE1BQU0sSUFBSSxLQUFLLENBQUM7WUFDUixRQUFRLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUMzQjtJQUNELElBQUksV0FBVyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDMUIsTUFBTSxJQUFJLEtBQUssQ0FBQztZQUNSLFdBQVcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzlCO0lBRUQsTUFBTSxNQUFNLEdBQTRCO1FBQ3RDLElBQUksRUFBRSxLQUFLO1FBQ1gsT0FBTyxFQUFFLFFBQVE7UUFDakIsVUFBVSxFQUFFLFdBQVc7S0FDeEIsQ0FBQztJQUVGLE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FBQyxpQkFBaUIsRUFBRSxNQUFZLENBQUMsQ0FBQztBQUMzRCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0saUJBQWlCLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLGtCQUFrQixFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uLy4uL2VuZ2luZSc7XG5pbXBvcnQge1NwYXJzZVNlZ21lbnRNZWFuLCBTcGFyc2VTZWdtZW50TWVhbklucHV0c30gZnJvbSAnLi4vLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7VGVuc29yLCBUZW5zb3IxRH0gZnJvbSAnLi4vLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi8uLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi8uLi90eXBlcyc7XG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBtZWFuIGFsb25nIHNwYXJzZSBzZWdtZW50cyBvZiBhIHRlbnNvci5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYyA9IHRmLnRlbnNvcjJkKFtbMSwyLDMsNF0sIFstMSwtMiwtMywtNF0sIFs2LDcsOCw5XV0pO1xuICogLy8gU2VsZWN0IHR3byByb3dzLCBvbmUgc2VnbWVudC5cbiAqIGNvbnN0IHJlc3VsdDEgPSB0Zi5zcGFyc2Uuc3BhcnNlU2VnbWVudE1lYW4oYyxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjFkKFswLCAxXSwgJ2ludDMyJyksXG4gKiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0Zi50ZW5zb3IxZChbMCwgMF0sICdpbnQzMicpKTtcbiAqIHJlc3VsdDEucHJpbnQoKTsgLy8gW1swLCAwLCAwLCAwXV1cbiAqXG4gKiAvLyBTZWxlY3QgdHdvIHJvd3MsIHR3byBzZWdtZW50cy5cbiAqIGNvbnN0IHJlc3VsdDIgPSB0Zi5zcGFyc2Uuc3BhcnNlU2VnbWVudE1lYW4oYyxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGYudGVuc29yMWQoWzAsIDFdLCAnaW50MzInKSxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGYudGVuc29yMWQoWzAsIDFdLCAnaW50MzInKSk7XG4gKiByZXN1bHQyLnByaW50KCk7IC8vIFtbMSwgMiwgMywgNF0sIFstMSwgLTIsIC0zLCAtNF1dXG4gKlxuICogLy8gU2VsZWN0IGFsbCByb3dzLCB0d28gc2VnbWVudHMuXG4gKiBjb25zdCByZXN1bHQzID0gdGYuc3BhcnNlLnNwYXJzZVNlZ21lbnRNZWFuKGMsXG4gKiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjFkKFswLCAxLCAyXSwgJ2ludDMyJyksXG4gKiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjFkKFswLCAxLCAxXSwgJ2ludDMyJykpO1xuICogcmVzdWx0My5wcmludCgpOyAvLyBbWzEuMCwgMi4wLCAzLjAsIDQuMF0sIFsyLjUsIDIuNSwgMi41LCAyLjVdXVxuICogYGBgXG4gKiBAcGFyYW0gZGF0YTogQSBUZW5zb3Igb2YgYXQgbGVhc3Qgb25lIGRpbWVuc2lvbiB3aXRoIGRhdGEgdGhhdCB3aWxsIGJlXG4gKiAgICAgYXNzZW1ibGVkIGluIHRoZSBvdXRwdXQuXG4gKiBAcGFyYW0gaW5kaWNlczogQSAxLUQgVGVuc29yIHdpdGggaW5kaWNlcyBpbnRvIGRhdGEuIEhhcyBzYW1lIHJhbmsgYXNcbiAqICAgICBzZWdtZW50SWRzLlxuICogQHBhcmFtIHNlZ21lbnRJZHM6IEEgMS1EIFRlbnNvciB3aXRoIGluZGljZXMgaW50byB0aGUgb3V0cHV0IFRlbnNvci4gVmFsdWVzXG4gKiAgICAgc2hvdWxkIGJlIHNvcnRlZCBhbmQgY2FuIGJlIHJlcGVhdGVkLlxuICogQHJldHVybiBIYXMgc2FtZSBzaGFwZSBhcyBkYXRhLCBleGNlcHQgZm9yIGRpbWVuc2lvbiAwIHdoaWNoIGhhcyBlcXVhbCB0b1xuICogICAgICAgICB0aGUgbnVtYmVyIG9mIHNlZ21lbnRzLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ1NwYXJzZSd9XG4gKi9cbmZ1bmN0aW9uIHNwYXJzZVNlZ21lbnRNZWFuXyhcbiAgICBkYXRhOiBUZW5zb3J8VGVuc29yTGlrZSwgaW5kaWNlczogVGVuc29yMUR8VGVuc29yTGlrZSxcbiAgICBzZWdtZW50SWRzOiBUZW5zb3IxRHxUZW5zb3JMaWtlKTogVGVuc29yIHtcbiAgY29uc3QgJGRhdGEgPSBjb252ZXJ0VG9UZW5zb3IoZGF0YSwgJ2RhdGEnLCAnc3BhcnNlU2VnbWVudE1lYW4nKTtcbiAgY29uc3QgJGluZGljZXMgPVxuICAgICAgY29udmVydFRvVGVuc29yKGluZGljZXMsICdpbmRpY2VzJywgJ3NwYXJzZVNlZ21lbnRNZWFuJywgJ2ludDMyJyk7XG4gIGNvbnN0ICRzZWdtZW50SWRzID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihzZWdtZW50SWRzLCAnc2VnbWVudElkcycsICdzcGFyc2VTZWdtZW50TWVhbicsICdpbnQzMicpO1xuXG4gIGlmICgkZGF0YS5yYW5rIDwgMSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYERhdGEgc2hvdWxkIGJlIGF0IGxlYXN0IDEgZGltZW5zaW9uYWwgYnV0IHJlY2VpdmVkIHNjYWxhcmApO1xuICB9XG4gIGlmICgkaW5kaWNlcy5yYW5rICE9PSAxKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBJbmRpY2VzIHNob3VsZCBiZSBUZW5zb3IxRCBidXQgcmVjZWl2ZWQgc2hhcGVcbiAgICAgICAgICAkeyRpbmRpY2VzLnNoYXBlfWApO1xuICB9XG4gIGlmICgkc2VnbWVudElkcy5yYW5rICE9PSAxKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBTZWdtZW50IGlkcyBzaG91bGQgYmUgVGVuc29yMUQgYnV0IHJlY2VpdmVkIHNoYXBlXG4gICAgICAgICAgJHskc2VnbWVudElkcy5zaGFwZX1gKTtcbiAgfVxuXG4gIGNvbnN0IGlucHV0czogU3BhcnNlU2VnbWVudE1lYW5JbnB1dHMgPSB7XG4gICAgZGF0YTogJGRhdGEsXG4gICAgaW5kaWNlczogJGluZGljZXMsXG4gICAgc2VnbWVudElkczogJHNlZ21lbnRJZHNcbiAgfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChTcGFyc2VTZWdtZW50TWVhbiwgaW5wdXRzIGFzIHt9KTtcbn1cblxuZXhwb3J0IGNvbnN0IHNwYXJzZVNlZ21lbnRNZWFuID0gLyogQF9fUFVSRV9fICovIG9wKHtzcGFyc2VTZWdtZW50TWVhbl99KTtcbiJdfQ==