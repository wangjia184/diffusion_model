/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { UnsortedSegmentSum } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { assert, isInt } from '../util';
import { op } from './operation';
/**
 * Computes the sum along segments of a `tf.Tensor`.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const segmentIds = tf.tensor1d([1, 2, 0, 1], 'int32');
 * const numSegments = 3;
 *
 * x.unsortedSegmentSum(segmentIds, numSegments).print()
 * //or tf.unsortedSegmentSum(x, segmentIds, numSegments)
 * ```
 * @param x The `tf.Tensor` that will be summed along its segments.
 * @param segmentIds A `tf.Tensor1D` whose rank is equal to the rank of `x`'s
 * dimension along the `axis`.  Maps each element of `x` to a segment.
 * @param numSegments The number of distinct `segmentIds`.
 *
 * @doc {heading: 'Operations', subheading: 'Segment'}
 */
function unsortedSegmentSum_(x, segmentIds, numSegments) {
    const $x = convertToTensor(x, 'x', 'unsortedSegmentSum');
    const $segmentIds = convertToTensor(segmentIds, 'segmentIds', 'unsortedSegmentSum', 'int32');
    assert(isInt(numSegments), () => 'numSegments must be of dtype int');
    const inputs = { x: $x, segmentIds: $segmentIds };
    const attrs = { numSegments };
    return ENGINE.runKernel(UnsortedSegmentSum, inputs, attrs);
}
export const unsortedSegmentSum = /* @__PURE__ */ op({ unsortedSegmentSum_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidW5zb3J0ZWRfc2VnbWVudF9zdW0uanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy91bnNvcnRlZF9zZWdtZW50X3N1bS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxrQkFBa0IsRUFBb0QsTUFBTSxpQkFBaUIsQ0FBQztBQUl0RyxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxFQUFDLE1BQU0sRUFBRSxLQUFLLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFdEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FpQkc7QUFDSCxTQUFTLG1CQUFtQixDQUN4QixDQUFlLEVBQUUsVUFBK0IsRUFBRSxXQUFtQjtJQUN2RSxNQUFNLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxvQkFBb0IsQ0FBQyxDQUFDO0lBQ3pELE1BQU0sV0FBVyxHQUNiLGVBQWUsQ0FBQyxVQUFVLEVBQUUsWUFBWSxFQUFFLG9CQUFvQixFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQzdFLE1BQU0sQ0FBQyxLQUFLLENBQUMsV0FBVyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUMsa0NBQWtDLENBQUMsQ0FBQztJQUVyRSxNQUFNLE1BQU0sR0FBNkIsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLFVBQVUsRUFBRSxXQUFXLEVBQUMsQ0FBQztJQUMxRSxNQUFNLEtBQUssR0FBNEIsRUFBQyxXQUFXLEVBQUMsQ0FBQztJQUVyRCxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLGtCQUFrQixFQUFFLE1BQW1DLEVBQ3ZELEtBQWdDLENBQUMsQ0FBQztBQUN4QyxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sa0JBQWtCLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLG1CQUFtQixFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1Vuc29ydGVkU2VnbWVudFN1bSwgVW5zb3J0ZWRTZWdtZW50U3VtQXR0cnMsIFVuc29ydGVkU2VnbWVudFN1bUlucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3IsIFRlbnNvcjFEfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2Fzc2VydCwgaXNJbnR9IGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIHN1bSBhbG9uZyBzZWdtZW50cyBvZiBhIGB0Zi5UZW5zb3JgLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDMsIDRdKTtcbiAqIGNvbnN0IHNlZ21lbnRJZHMgPSB0Zi50ZW5zb3IxZChbMSwgMiwgMCwgMV0sICdpbnQzMicpO1xuICogY29uc3QgbnVtU2VnbWVudHMgPSAzO1xuICpcbiAqIHgudW5zb3J0ZWRTZWdtZW50U3VtKHNlZ21lbnRJZHMsIG51bVNlZ21lbnRzKS5wcmludCgpXG4gKiAvL29yIHRmLnVuc29ydGVkU2VnbWVudFN1bSh4LCBzZWdtZW50SWRzLCBudW1TZWdtZW50cylcbiAqIGBgYFxuICogQHBhcmFtIHggVGhlIGB0Zi5UZW5zb3JgIHRoYXQgd2lsbCBiZSBzdW1tZWQgYWxvbmcgaXRzIHNlZ21lbnRzLlxuICogQHBhcmFtIHNlZ21lbnRJZHMgQSBgdGYuVGVuc29yMURgIHdob3NlIHJhbmsgaXMgZXF1YWwgdG8gdGhlIHJhbmsgb2YgYHhgJ3NcbiAqIGRpbWVuc2lvbiBhbG9uZyB0aGUgYGF4aXNgLiAgTWFwcyBlYWNoIGVsZW1lbnQgb2YgYHhgIHRvIGEgc2VnbWVudC5cbiAqIEBwYXJhbSBudW1TZWdtZW50cyBUaGUgbnVtYmVyIG9mIGRpc3RpbmN0IGBzZWdtZW50SWRzYC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdTZWdtZW50J31cbiAqL1xuZnVuY3Rpb24gdW5zb3J0ZWRTZWdtZW50U3VtXzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICB4OiBUfFRlbnNvckxpa2UsIHNlZ21lbnRJZHM6IFRlbnNvcjFEfFRlbnNvckxpa2UsIG51bVNlZ21lbnRzOiBudW1iZXIpOiBUIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAndW5zb3J0ZWRTZWdtZW50U3VtJyk7XG4gIGNvbnN0ICRzZWdtZW50SWRzID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihzZWdtZW50SWRzLCAnc2VnbWVudElkcycsICd1bnNvcnRlZFNlZ21lbnRTdW0nLCAnaW50MzInKTtcbiAgYXNzZXJ0KGlzSW50KG51bVNlZ21lbnRzKSwgKCkgPT4gJ251bVNlZ21lbnRzIG11c3QgYmUgb2YgZHR5cGUgaW50Jyk7XG5cbiAgY29uc3QgaW5wdXRzOiBVbnNvcnRlZFNlZ21lbnRTdW1JbnB1dHMgPSB7eDogJHgsIHNlZ21lbnRJZHM6ICRzZWdtZW50SWRzfTtcbiAgY29uc3QgYXR0cnM6IFVuc29ydGVkU2VnbWVudFN1bUF0dHJzID0ge251bVNlZ21lbnRzfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIFVuc29ydGVkU2VnbWVudFN1bSwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCB1bnNvcnRlZFNlZ21lbnRTdW0gPSAvKiBAX19QVVJFX18gKi8gb3Aoe3Vuc29ydGVkU2VnbWVudFN1bV99KTtcbiJdfQ==