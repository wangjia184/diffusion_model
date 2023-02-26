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
import { SearchSorted } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { sizeFromShape } from '../util_base';
import { op } from './operation';
import { reshape } from './reshape';
const INT32_MAX = 2147483648;
/**
 * Searches for where a value would go in a sorted sequence.
 *
 * This is not a method for checking containment (like javascript in).
 *
 * The typical use case for this operation is "binning", "bucketing", or
 * "discretizing". The values are assigned to bucket-indices based on the edges
 * listed in 'sortedSequence'. This operation returns the bucket-index for each
 * value.
 *
 * The side argument controls which index is returned if a value lands exactly
 * on an edge.
 *
 * The axis is not settable for this operation. It always operates on the
 * innermost dimension (axis=-1). The operation will accept any number of outer
 * dimensions.
 *
 * Note: This operation assumes that 'sortedSequence' is sorted along the
 * innermost axis, maybe using 'sort(..., axis=-1)'. If the sequence is not
 * sorted no error is raised and the content of the returned tensor is not well
 * defined.
 *
 * ```js
 * const edges = tf.tensor1d([-1, 3.3, 9.1, 10.0]);
 * let values = tf.tensor1d([0.0, 4.1, 12.0]);
 * const result1 = tf.searchSorted(edges, values, 'left');
 * result1.print(); // [1, 2, 4]
 *
 * const seq = tf.tensor1d([0, 3, 9, 10, 10]);
 * values = tf.tensor1d([0, 4, 10]);
 * const result2 = tf.searchSorted(seq, values, 'left');
 * result2.print(); // [0, 2, 3]
 * const result3 = tf.searchSorted(seq, values, 'right');
 * result3.print(); // [1, 2, 5]
 *
 * const sortedSequence = tf.tensor2d([[0., 3., 8., 9., 10.],
 *                                     [1., 2., 3., 4., 5.]]);
 * values = tf.tensor2d([[9.8, 2.1, 4.3],
 *                       [0.1, 6.6, 4.5, ]]);
 * const result4 = tf.searchSorted(sortedSequence, values, 'left');
 * result4.print(); // [[4, 1, 2], [0, 5, 4]]
 * ```
 * @param sortedSequence: N-D. Sorted sequence.
 * @param values: N-D. Search values.
 * @param side: 'left'|'right'. Defaults to 'left'. 'left' corresponds to lower
 *     bound and 'right' to upper bound.
 * @return An N-D int32 tensor the size of values containing the result of
 *     applying either lower bound or upper bound (depending on side) to each
 *     value. The result is not a global index to the entire Tensor, but the
 *     index in the last dimension.
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
function searchSorted_(sortedSequence, values, side = 'left') {
    const $sortedSequence = convertToTensor(sortedSequence, 'sortedSequence', 'searchSorted');
    const $values = convertToTensor(values, 'values', 'searchSorted');
    const sequenceSize = $sortedSequence.shape[$sortedSequence.shape.length - 1];
    const valuesSize = $values.shape[$values.shape.length - 1];
    const $sortedSequence2D = reshape($sortedSequence, [-1, sequenceSize]);
    const $values2D = reshape($values, [-1, valuesSize]);
    if ($sortedSequence2D.rank < 2) {
        throw new Error(`Sorted input argument must be at least 2-dimensional`);
    }
    if ($sortedSequence2D.shape[0] !== $values2D.shape[0]) {
        throw new Error(`Leading dimension of 'sortedSequence' and 'values' must match.`);
    }
    if (sizeFromShape($values2D.shape) >= INT32_MAX) {
        throw new Error(`values tensor size must less than ${INT32_MAX}`);
    }
    if ($sortedSequence2D.shape[1] >= INT32_MAX) {
        throw new Error(`trailing dim_size must less than ${INT32_MAX} for int32 output type, was ${$sortedSequence2D.shape[1]}`);
    }
    const inputs = {
        sortedSequence: $sortedSequence2D,
        values: $values2D,
    };
    const attrs = { side };
    return ENGINE.runKernel(SearchSorted, inputs, attrs);
}
export const searchSorted = /* @__PURE__ */ op({ searchSorted_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2VhcmNoX3NvcnRlZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NlYXJjaF9zb3J0ZWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsWUFBWSxFQUF3QyxNQUFNLGlCQUFpQixDQUFDO0FBRXBGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEVBQUMsYUFBYSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQzNDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUVsQyxNQUFNLFNBQVMsR0FBRyxVQUFVLENBQUM7QUFDN0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQW1ERztBQUNILFNBQVMsYUFBYSxDQUNsQixjQUFpQyxFQUFFLE1BQXlCLEVBQzVELE9BQXVCLE1BQU07SUFDL0IsTUFBTSxlQUFlLEdBQ2pCLGVBQWUsQ0FBQyxjQUFjLEVBQUUsZ0JBQWdCLEVBQUUsY0FBYyxDQUFDLENBQUM7SUFDdEUsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsY0FBYyxDQUFDLENBQUM7SUFFbEUsTUFBTSxZQUFZLEdBQUcsZUFBZSxDQUFDLEtBQUssQ0FBQyxlQUFlLENBQUMsS0FBSyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQztJQUM3RSxNQUFNLFVBQVUsR0FBRyxPQUFPLENBQUMsS0FBSyxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBQzNELE1BQU0saUJBQWlCLEdBQUcsT0FBTyxDQUFDLGVBQWUsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLFlBQVksQ0FBQyxDQUFDLENBQUM7SUFDdkUsTUFBTSxTQUFTLEdBQUcsT0FBTyxDQUFDLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLFVBQVUsQ0FBQyxDQUFDLENBQUM7SUFFckQsSUFBSSxpQkFBaUIsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxFQUFFO1FBQzlCLE1BQU0sSUFBSSxLQUFLLENBQUMsc0RBQXNELENBQUMsQ0FBQztLQUN6RTtJQUNELElBQUksaUJBQWlCLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLFNBQVMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUU7UUFDckQsTUFBTSxJQUFJLEtBQUssQ0FDWCxnRUFBZ0UsQ0FBQyxDQUFDO0tBQ3ZFO0lBQ0QsSUFBSSxhQUFhLENBQUMsU0FBUyxDQUFDLEtBQUssQ0FBQyxJQUFJLFNBQVMsRUFBRTtRQUMvQyxNQUFNLElBQUksS0FBSyxDQUFDLHFDQUFxQyxTQUFTLEVBQUUsQ0FBQyxDQUFDO0tBQ25FO0lBQ0QsSUFBSSxpQkFBaUIsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLElBQUksU0FBUyxFQUFFO1FBQzNDLE1BQU0sSUFBSSxLQUFLLENBQUMsb0NBQ1osU0FBUywrQkFBK0IsaUJBQWlCLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztLQUMzRTtJQUVELE1BQU0sTUFBTSxHQUF1QjtRQUNqQyxjQUFjLEVBQUUsaUJBQWlCO1FBQ2pDLE1BQU0sRUFBRSxTQUFTO0tBQ2xCLENBQUM7SUFDRixNQUFNLEtBQUssR0FBc0IsRUFBQyxJQUFJLEVBQUMsQ0FBQztJQUV4QyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQUMsWUFBWSxFQUFFLE1BQVksRUFBRSxLQUFXLENBQUMsQ0FBQztBQUNuRSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sWUFBWSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxhQUFhLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7U2VhcmNoU29ydGVkLCBTZWFyY2hTb3J0ZWRBdHRycywgU2VhcmNoU29ydGVkSW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHtzaXplRnJvbVNoYXBlfSBmcm9tICcuLi91dGlsX2Jhc2UnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL3Jlc2hhcGUnO1xuXG5jb25zdCBJTlQzMl9NQVggPSAyMTQ3NDgzNjQ4O1xuLyoqXG4gKiBTZWFyY2hlcyBmb3Igd2hlcmUgYSB2YWx1ZSB3b3VsZCBnbyBpbiBhIHNvcnRlZCBzZXF1ZW5jZS5cbiAqXG4gKiBUaGlzIGlzIG5vdCBhIG1ldGhvZCBmb3IgY2hlY2tpbmcgY29udGFpbm1lbnQgKGxpa2UgamF2YXNjcmlwdCBpbikuXG4gKlxuICogVGhlIHR5cGljYWwgdXNlIGNhc2UgZm9yIHRoaXMgb3BlcmF0aW9uIGlzIFwiYmlubmluZ1wiLCBcImJ1Y2tldGluZ1wiLCBvclxuICogXCJkaXNjcmV0aXppbmdcIi4gVGhlIHZhbHVlcyBhcmUgYXNzaWduZWQgdG8gYnVja2V0LWluZGljZXMgYmFzZWQgb24gdGhlIGVkZ2VzXG4gKiBsaXN0ZWQgaW4gJ3NvcnRlZFNlcXVlbmNlJy4gVGhpcyBvcGVyYXRpb24gcmV0dXJucyB0aGUgYnVja2V0LWluZGV4IGZvciBlYWNoXG4gKiB2YWx1ZS5cbiAqXG4gKiBUaGUgc2lkZSBhcmd1bWVudCBjb250cm9scyB3aGljaCBpbmRleCBpcyByZXR1cm5lZCBpZiBhIHZhbHVlIGxhbmRzIGV4YWN0bHlcbiAqIG9uIGFuIGVkZ2UuXG4gKlxuICogVGhlIGF4aXMgaXMgbm90IHNldHRhYmxlIGZvciB0aGlzIG9wZXJhdGlvbi4gSXQgYWx3YXlzIG9wZXJhdGVzIG9uIHRoZVxuICogaW5uZXJtb3N0IGRpbWVuc2lvbiAoYXhpcz0tMSkuIFRoZSBvcGVyYXRpb24gd2lsbCBhY2NlcHQgYW55IG51bWJlciBvZiBvdXRlclxuICogZGltZW5zaW9ucy5cbiAqXG4gKiBOb3RlOiBUaGlzIG9wZXJhdGlvbiBhc3N1bWVzIHRoYXQgJ3NvcnRlZFNlcXVlbmNlJyBpcyBzb3J0ZWQgYWxvbmcgdGhlXG4gKiBpbm5lcm1vc3QgYXhpcywgbWF5YmUgdXNpbmcgJ3NvcnQoLi4uLCBheGlzPS0xKScuIElmIHRoZSBzZXF1ZW5jZSBpcyBub3RcbiAqIHNvcnRlZCBubyBlcnJvciBpcyByYWlzZWQgYW5kIHRoZSBjb250ZW50IG9mIHRoZSByZXR1cm5lZCB0ZW5zb3IgaXMgbm90IHdlbGxcbiAqIGRlZmluZWQuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGVkZ2VzID0gdGYudGVuc29yMWQoWy0xLCAzLjMsIDkuMSwgMTAuMF0pO1xuICogbGV0IHZhbHVlcyA9IHRmLnRlbnNvcjFkKFswLjAsIDQuMSwgMTIuMF0pO1xuICogY29uc3QgcmVzdWx0MSA9IHRmLnNlYXJjaFNvcnRlZChlZGdlcywgdmFsdWVzLCAnbGVmdCcpO1xuICogcmVzdWx0MS5wcmludCgpOyAvLyBbMSwgMiwgNF1cbiAqXG4gKiBjb25zdCBzZXEgPSB0Zi50ZW5zb3IxZChbMCwgMywgOSwgMTAsIDEwXSk7XG4gKiB2YWx1ZXMgPSB0Zi50ZW5zb3IxZChbMCwgNCwgMTBdKTtcbiAqIGNvbnN0IHJlc3VsdDIgPSB0Zi5zZWFyY2hTb3J0ZWQoc2VxLCB2YWx1ZXMsICdsZWZ0Jyk7XG4gKiByZXN1bHQyLnByaW50KCk7IC8vIFswLCAyLCAzXVxuICogY29uc3QgcmVzdWx0MyA9IHRmLnNlYXJjaFNvcnRlZChzZXEsIHZhbHVlcywgJ3JpZ2h0Jyk7XG4gKiByZXN1bHQzLnByaW50KCk7IC8vIFsxLCAyLCA1XVxuICpcbiAqIGNvbnN0IHNvcnRlZFNlcXVlbmNlID0gdGYudGVuc29yMmQoW1swLiwgMy4sIDguLCA5LiwgMTAuXSxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFsxLiwgMi4sIDMuLCA0LiwgNS5dXSk7XG4gKiB2YWx1ZXMgPSB0Zi50ZW5zb3IyZChbWzkuOCwgMi4xLCA0LjNdLFxuICogICAgICAgICAgICAgICAgICAgICAgIFswLjEsIDYuNiwgNC41LCBdXSk7XG4gKiBjb25zdCByZXN1bHQ0ID0gdGYuc2VhcmNoU29ydGVkKHNvcnRlZFNlcXVlbmNlLCB2YWx1ZXMsICdsZWZ0Jyk7XG4gKiByZXN1bHQ0LnByaW50KCk7IC8vIFtbNCwgMSwgMl0sIFswLCA1LCA0XV1cbiAqIGBgYFxuICogQHBhcmFtIHNvcnRlZFNlcXVlbmNlOiBOLUQuIFNvcnRlZCBzZXF1ZW5jZS5cbiAqIEBwYXJhbSB2YWx1ZXM6IE4tRC4gU2VhcmNoIHZhbHVlcy5cbiAqIEBwYXJhbSBzaWRlOiAnbGVmdCd8J3JpZ2h0Jy4gRGVmYXVsdHMgdG8gJ2xlZnQnLiAnbGVmdCcgY29ycmVzcG9uZHMgdG8gbG93ZXJcbiAqICAgICBib3VuZCBhbmQgJ3JpZ2h0JyB0byB1cHBlciBib3VuZC5cbiAqIEByZXR1cm4gQW4gTi1EIGludDMyIHRlbnNvciB0aGUgc2l6ZSBvZiB2YWx1ZXMgY29udGFpbmluZyB0aGUgcmVzdWx0IG9mXG4gKiAgICAgYXBwbHlpbmcgZWl0aGVyIGxvd2VyIGJvdW5kIG9yIHVwcGVyIGJvdW5kIChkZXBlbmRpbmcgb24gc2lkZSkgdG8gZWFjaFxuICogICAgIHZhbHVlLiBUaGUgcmVzdWx0IGlzIG5vdCBhIGdsb2JhbCBpbmRleCB0byB0aGUgZW50aXJlIFRlbnNvciwgYnV0IHRoZVxuICogICAgIGluZGV4IGluIHRoZSBsYXN0IGRpbWVuc2lvbi5cbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ0V2YWx1YXRpb24nfVxuICovXG5mdW5jdGlvbiBzZWFyY2hTb3J0ZWRfKFxuICAgIHNvcnRlZFNlcXVlbmNlOiBUZW5zb3J8VGVuc29yTGlrZSwgdmFsdWVzOiBUZW5zb3J8VGVuc29yTGlrZSxcbiAgICBzaWRlOiAnbGVmdCd8J3JpZ2h0JyA9ICdsZWZ0Jyk6IFRlbnNvciB7XG4gIGNvbnN0ICRzb3J0ZWRTZXF1ZW5jZSA9XG4gICAgICBjb252ZXJ0VG9UZW5zb3Ioc29ydGVkU2VxdWVuY2UsICdzb3J0ZWRTZXF1ZW5jZScsICdzZWFyY2hTb3J0ZWQnKTtcbiAgY29uc3QgJHZhbHVlcyA9IGNvbnZlcnRUb1RlbnNvcih2YWx1ZXMsICd2YWx1ZXMnLCAnc2VhcmNoU29ydGVkJyk7XG5cbiAgY29uc3Qgc2VxdWVuY2VTaXplID0gJHNvcnRlZFNlcXVlbmNlLnNoYXBlWyRzb3J0ZWRTZXF1ZW5jZS5zaGFwZS5sZW5ndGggLSAxXTtcbiAgY29uc3QgdmFsdWVzU2l6ZSA9ICR2YWx1ZXMuc2hhcGVbJHZhbHVlcy5zaGFwZS5sZW5ndGggLSAxXTtcbiAgY29uc3QgJHNvcnRlZFNlcXVlbmNlMkQgPSByZXNoYXBlKCRzb3J0ZWRTZXF1ZW5jZSwgWy0xLCBzZXF1ZW5jZVNpemVdKTtcbiAgY29uc3QgJHZhbHVlczJEID0gcmVzaGFwZSgkdmFsdWVzLCBbLTEsIHZhbHVlc1NpemVdKTtcblxuICBpZiAoJHNvcnRlZFNlcXVlbmNlMkQucmFuayA8IDIpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFNvcnRlZCBpbnB1dCBhcmd1bWVudCBtdXN0IGJlIGF0IGxlYXN0IDItZGltZW5zaW9uYWxgKTtcbiAgfVxuICBpZiAoJHNvcnRlZFNlcXVlbmNlMkQuc2hhcGVbMF0gIT09ICR2YWx1ZXMyRC5zaGFwZVswXSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYExlYWRpbmcgZGltZW5zaW9uIG9mICdzb3J0ZWRTZXF1ZW5jZScgYW5kICd2YWx1ZXMnIG11c3QgbWF0Y2guYCk7XG4gIH1cbiAgaWYgKHNpemVGcm9tU2hhcGUoJHZhbHVlczJELnNoYXBlKSA+PSBJTlQzMl9NQVgpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYHZhbHVlcyB0ZW5zb3Igc2l6ZSBtdXN0IGxlc3MgdGhhbiAke0lOVDMyX01BWH1gKTtcbiAgfVxuICBpZiAoJHNvcnRlZFNlcXVlbmNlMkQuc2hhcGVbMV0gPj0gSU5UMzJfTUFYKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGB0cmFpbGluZyBkaW1fc2l6ZSBtdXN0IGxlc3MgdGhhbiAke1xuICAgICAgICBJTlQzMl9NQVh9IGZvciBpbnQzMiBvdXRwdXQgdHlwZSwgd2FzICR7JHNvcnRlZFNlcXVlbmNlMkQuc2hhcGVbMV19YCk7XG4gIH1cblxuICBjb25zdCBpbnB1dHM6IFNlYXJjaFNvcnRlZElucHV0cyA9IHtcbiAgICBzb3J0ZWRTZXF1ZW5jZTogJHNvcnRlZFNlcXVlbmNlMkQsXG4gICAgdmFsdWVzOiAkdmFsdWVzMkQsXG4gIH07XG4gIGNvbnN0IGF0dHJzOiBTZWFyY2hTb3J0ZWRBdHRycyA9IHtzaWRlfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChTZWFyY2hTb3J0ZWQsIGlucHV0cyBhcyB7fSwgYXR0cnMgYXMge30pO1xufVxuXG5leHBvcnQgY29uc3Qgc2VhcmNoU29ydGVkID0gLyogQF9fUFVSRV9fICovIG9wKHtzZWFyY2hTb3J0ZWRffSk7XG4iXX0=