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
import { ENGINE } from '../engine';
import { StridedSlice } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Extracts a strided slice of a tensor.
 *
 * Roughly speaking, this op extracts a slice of size (end-begin)/stride from
 * the given input tensor (x). Starting at the location specified by begin the
 * slice continues by adding stride to the index until all dimensions are not
 * less than end. Note that a stride can be negative, which causes a reverse
 * slice.
 *
 * ```js
 * const t = tf.tensor3d([1, 1, 1 ,2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
 *    [3, 2, 3]);
 * t.stridedSlice([1, 0, 0], [2, 1, 3], [1, 1, 1]).print()  // [[[3, 3, 3]]]
 * t.stridedSlice([1, 0, 0], [2, 2, 3], [1, 1, 1]).print()  // [[[3, 3, 3],
 *                                                     // [4, 4, 4]]]
 * t.stridedSlice([1, -1, 0], [2, -3, 3], [1, -1, 1]).print() // [[[4, 4, 4],
 *                                                     // [3, 3, 3]]]
 * ```
 *
 * @param x The tensor to stride slice.
 * @param begin The coordinates to start the slice from.
 * @param end: The coordinates to end the slice at.
 * @param strides: The size of the slice.
 * @param beginMask: If the ith bit of beginMask is set, begin[i] is ignored
 *      and the fullest possible range in that dimension is used instead.
 * @param endMask: If the ith bit of endMask is set, end[i] is ignored
 *      and the fullest possible range in that dimension is used instead.
 * @param shrinkAxisMask: a bitmask where bit i implies that
 * the ith specification should shrink the dimensionality. begin and end must
 * imply a slice of size 1 in the dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function stridedSlice_(x, begin, end, strides, beginMask = 0, endMask = 0, ellipsisMask = 0, newAxisMask = 0, shrinkAxisMask = 0) {
    const $x = convertToTensor(x, 'x', 'stridedSlice', 'string_or_numeric');
    const inputs = { x: $x };
    const attrs = {
        begin,
        end,
        strides,
        beginMask,
        endMask,
        ellipsisMask,
        newAxisMask,
        shrinkAxisMask
    };
    return ENGINE.runKernel(StridedSlice, inputs, attrs);
}
export const stridedSlice = /* @__PURE__ */ op({ stridedSlice_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3RyaWRlZF9zbGljZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3N0cmlkZWRfc2xpY2UudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsWUFBWSxFQUF3QyxNQUFNLGlCQUFpQixDQUFDO0FBSXBGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQWdDRztBQUNILFNBQVMsYUFBYSxDQUNsQixDQUFvQixFQUFFLEtBQWUsRUFBRSxHQUFhLEVBQUUsT0FBa0IsRUFDeEUsU0FBUyxHQUFHLENBQUMsRUFBRSxPQUFPLEdBQUcsQ0FBQyxFQUFFLFlBQVksR0FBRyxDQUFDLEVBQUUsV0FBVyxHQUFHLENBQUMsRUFDN0QsY0FBYyxHQUFHLENBQUM7SUFDcEIsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsY0FBYyxFQUFFLG1CQUFtQixDQUFDLENBQUM7SUFFeEUsTUFBTSxNQUFNLEdBQXVCLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQzNDLE1BQU0sS0FBSyxHQUFzQjtRQUMvQixLQUFLO1FBQ0wsR0FBRztRQUNILE9BQU87UUFDUCxTQUFTO1FBQ1QsT0FBTztRQUNQLFlBQVk7UUFDWixXQUFXO1FBQ1gsY0FBYztLQUNmLENBQUM7SUFFRixPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLFlBQVksRUFBRSxNQUFtQyxFQUNqRCxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFlBQVksR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsYUFBYSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1N0cmlkZWRTbGljZSwgU3RyaWRlZFNsaWNlQXR0cnMsIFN0cmlkZWRTbGljZUlucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIEV4dHJhY3RzIGEgc3RyaWRlZCBzbGljZSBvZiBhIHRlbnNvci5cbiAqXG4gKiBSb3VnaGx5IHNwZWFraW5nLCB0aGlzIG9wIGV4dHJhY3RzIGEgc2xpY2Ugb2Ygc2l6ZSAoZW5kLWJlZ2luKS9zdHJpZGUgZnJvbVxuICogdGhlIGdpdmVuIGlucHV0IHRlbnNvciAoeCkuIFN0YXJ0aW5nIGF0IHRoZSBsb2NhdGlvbiBzcGVjaWZpZWQgYnkgYmVnaW4gdGhlXG4gKiBzbGljZSBjb250aW51ZXMgYnkgYWRkaW5nIHN0cmlkZSB0byB0aGUgaW5kZXggdW50aWwgYWxsIGRpbWVuc2lvbnMgYXJlIG5vdFxuICogbGVzcyB0aGFuIGVuZC4gTm90ZSB0aGF0IGEgc3RyaWRlIGNhbiBiZSBuZWdhdGl2ZSwgd2hpY2ggY2F1c2VzIGEgcmV2ZXJzZVxuICogc2xpY2UuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHQgPSB0Zi50ZW5zb3IzZChbMSwgMSwgMSAsMiwgMiwgMiwgMywgMywgMywgNCwgNCwgNCwgNSwgNSwgNSwgNiwgNiwgNl0sXG4gKiAgICBbMywgMiwgM10pO1xuICogdC5zdHJpZGVkU2xpY2UoWzEsIDAsIDBdLCBbMiwgMSwgM10sIFsxLCAxLCAxXSkucHJpbnQoKSAgLy8gW1tbMywgMywgM11dXVxuICogdC5zdHJpZGVkU2xpY2UoWzEsIDAsIDBdLCBbMiwgMiwgM10sIFsxLCAxLCAxXSkucHJpbnQoKSAgLy8gW1tbMywgMywgM10sXG4gKiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gWzQsIDQsIDRdXV1cbiAqIHQuc3RyaWRlZFNsaWNlKFsxLCAtMSwgMF0sIFsyLCAtMywgM10sIFsxLCAtMSwgMV0pLnByaW50KCkgLy8gW1tbNCwgNCwgNF0sXG4gKiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gWzMsIDMsIDNdXV1cbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4IFRoZSB0ZW5zb3IgdG8gc3RyaWRlIHNsaWNlLlxuICogQHBhcmFtIGJlZ2luIFRoZSBjb29yZGluYXRlcyB0byBzdGFydCB0aGUgc2xpY2UgZnJvbS5cbiAqIEBwYXJhbSBlbmQ6IFRoZSBjb29yZGluYXRlcyB0byBlbmQgdGhlIHNsaWNlIGF0LlxuICogQHBhcmFtIHN0cmlkZXM6IFRoZSBzaXplIG9mIHRoZSBzbGljZS5cbiAqIEBwYXJhbSBiZWdpbk1hc2s6IElmIHRoZSBpdGggYml0IG9mIGJlZ2luTWFzayBpcyBzZXQsIGJlZ2luW2ldIGlzIGlnbm9yZWRcbiAqICAgICAgYW5kIHRoZSBmdWxsZXN0IHBvc3NpYmxlIHJhbmdlIGluIHRoYXQgZGltZW5zaW9uIGlzIHVzZWQgaW5zdGVhZC5cbiAqIEBwYXJhbSBlbmRNYXNrOiBJZiB0aGUgaXRoIGJpdCBvZiBlbmRNYXNrIGlzIHNldCwgZW5kW2ldIGlzIGlnbm9yZWRcbiAqICAgICAgYW5kIHRoZSBmdWxsZXN0IHBvc3NpYmxlIHJhbmdlIGluIHRoYXQgZGltZW5zaW9uIGlzIHVzZWQgaW5zdGVhZC5cbiAqIEBwYXJhbSBzaHJpbmtBeGlzTWFzazogYSBiaXRtYXNrIHdoZXJlIGJpdCBpIGltcGxpZXMgdGhhdFxuICogdGhlIGl0aCBzcGVjaWZpY2F0aW9uIHNob3VsZCBzaHJpbmsgdGhlIGRpbWVuc2lvbmFsaXR5LiBiZWdpbiBhbmQgZW5kIG11c3RcbiAqIGltcGx5IGEgc2xpY2Ugb2Ygc2l6ZSAxIGluIHRoZSBkaW1lbnNpb24uXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnU2xpY2luZyBhbmQgSm9pbmluZyd9XG4gKi9cbmZ1bmN0aW9uIHN0cmlkZWRTbGljZV8oXG4gICAgeDogVGVuc29yfFRlbnNvckxpa2UsIGJlZ2luOiBudW1iZXJbXSwgZW5kOiBudW1iZXJbXSwgc3RyaWRlcz86IG51bWJlcltdLFxuICAgIGJlZ2luTWFzayA9IDAsIGVuZE1hc2sgPSAwLCBlbGxpcHNpc01hc2sgPSAwLCBuZXdBeGlzTWFzayA9IDAsXG4gICAgc2hyaW5rQXhpc01hc2sgPSAwKTogVGVuc29yIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnc3RyaWRlZFNsaWNlJywgJ3N0cmluZ19vcl9udW1lcmljJyk7XG5cbiAgY29uc3QgaW5wdXRzOiBTdHJpZGVkU2xpY2VJbnB1dHMgPSB7eDogJHh9O1xuICBjb25zdCBhdHRyczogU3RyaWRlZFNsaWNlQXR0cnMgPSB7XG4gICAgYmVnaW4sXG4gICAgZW5kLFxuICAgIHN0cmlkZXMsXG4gICAgYmVnaW5NYXNrLFxuICAgIGVuZE1hc2ssXG4gICAgZWxsaXBzaXNNYXNrLFxuICAgIG5ld0F4aXNNYXNrLFxuICAgIHNocmlua0F4aXNNYXNrXG4gIH07XG5cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBTdHJpZGVkU2xpY2UsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3Qgc3RyaWRlZFNsaWNlID0gLyogQF9fUFVSRV9fICovIG9wKHtzdHJpZGVkU2xpY2VffSk7XG4iXX0=