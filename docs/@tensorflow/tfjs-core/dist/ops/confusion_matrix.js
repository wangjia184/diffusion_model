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
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { cast } from './cast';
import { matMul } from './mat_mul';
import { oneHot } from './one_hot';
import { op } from './operation';
import { transpose } from './transpose';
/**
 * Computes the confusion matrix from true labels and predicted labels.
 *
 * ```js
 * const labels = tf.tensor1d([0, 1, 2, 1, 0], 'int32');
 * const predictions = tf.tensor1d([0, 2, 2, 1, 0], 'int32');
 * const numClasses = 3;
 * const out = tf.math.confusionMatrix(labels, predictions, numClasses);
 * out.print();
 * // Expected output matrix:
 * // [[2, 0, 0],
 * //  [0, 1, 1],
 * //  [0, 0, 1]]
 * ```
 *
 * @param labels The target labels, assumed to be 0-based integers
 *   for the classes. The shape is `[numExamples]`, where
 *   `numExamples` is the number of examples included.
 * @param predictions The predicted classes, assumed to be
 *   0-based integers for the classes. Must have the same shape as `labels`.
 * @param numClasses Number of all classes, as an integer.
 *   Its value must be larger than the largest element in `labels` and
 *   `predictions`.
 * @returns The confusion matrix as a int32-type 2D tensor. The value at
 *   row `r` and column `c` is the number of times examples of actual class
 *   `r` were predicted as class `c`.
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
export function confusionMatrix_(labels, predictions, numClasses) {
    const $labels = convertToTensor(labels, 'labels', 'confusionMatrix');
    const $predictions = convertToTensor(predictions, 'predictions', 'confusionMatrix');
    util.assert(numClasses == null || numClasses > 0 && Number.isInteger(numClasses), () => `If provided, numClasses must be a positive integer, ` +
        `but got ${numClasses}`);
    util.assert($labels.rank === 1, () => `Expected the rank of labels to be 1, but got ${$labels.rank}`);
    util.assert($predictions.rank === 1, () => `Expected the rank of predictions to be 1, ` +
        `but got ${$predictions.rank}`);
    util.assert($labels.shape[0] === $predictions.shape[0], () => `Mismatch in the number of examples: ` +
        `${$labels.shape[0]} vs. ${$predictions.shape[0]}. ` +
        `Labels and predictions should have the same number of elements.`);
    util.assert(numClasses > 0 && Number.isInteger(numClasses), () => `numClasses is required to be a positive integer, but got ` +
        `${numClasses}`);
    // TODO(cais): In the future, if oneHot supports tensors inputs for
    //   `numClasses`, `confusionMatrix` can make `numClasses` optional.
    const oneHotLabels = oneHot(cast($labels, 'int32'), numClasses);
    const oneHotPredictions = oneHot(cast($predictions, 'int32'), numClasses);
    const oneHotLabelsT = transpose(oneHotLabels);
    const product = matMul(oneHotLabelsT, oneHotPredictions);
    return cast(product, 'int32');
}
export const confusionMatrix = /* @__PURE__ */ op({ confusionMatrix_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29uZnVzaW9uX21hdHJpeC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2NvbmZ1c2lvbl9tYXRyaXgudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDNUIsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUV0Qzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTRCRztBQUNILE1BQU0sVUFBVSxnQkFBZ0IsQ0FDNUIsTUFBMkIsRUFBRSxXQUFnQyxFQUM3RCxVQUFrQjtJQUNwQixNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxpQkFBaUIsQ0FBQyxDQUFDO0lBQ3JFLE1BQU0sWUFBWSxHQUNkLGVBQWUsQ0FBQyxXQUFXLEVBQUUsYUFBYSxFQUFFLGlCQUFpQixDQUFDLENBQUM7SUFFbkUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxVQUFVLElBQUksSUFBSSxJQUFJLFVBQVUsR0FBRyxDQUFDLElBQUksTUFBTSxDQUFDLFNBQVMsQ0FBQyxVQUFVLENBQUMsRUFDcEUsR0FBRyxFQUFFLENBQUMsc0RBQXNEO1FBQ3hELFdBQVcsVUFBVSxFQUFFLENBQUMsQ0FBQztJQUNqQyxJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNsQixHQUFHLEVBQUUsQ0FBQyxnREFBZ0QsT0FBTyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUM7SUFDMUUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxZQUFZLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDdkIsR0FBRyxFQUFFLENBQUMsNENBQTRDO1FBQzlDLFdBQVcsWUFBWSxDQUFDLElBQUksRUFBRSxDQUFDLENBQUM7SUFDeEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLFlBQVksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQzFDLEdBQUcsRUFBRSxDQUFDLHNDQUFzQztRQUN4QyxHQUFHLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLFFBQVEsWUFBWSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsSUFBSTtRQUNwRCxpRUFBaUUsQ0FBQyxDQUFDO0lBQzNFLElBQUksQ0FBQyxNQUFNLENBQ1AsVUFBVSxHQUFHLENBQUMsSUFBSSxNQUFNLENBQUMsU0FBUyxDQUFDLFVBQVUsQ0FBQyxFQUM5QyxHQUFHLEVBQUUsQ0FBQywyREFBMkQ7UUFDN0QsR0FBRyxVQUFVLEVBQUUsQ0FBQyxDQUFDO0lBQ3pCLG1FQUFtRTtJQUNuRSxvRUFBb0U7SUFFcEUsTUFBTSxZQUFZLEdBQUcsTUFBTSxDQUFDLElBQUksQ0FBQyxPQUFPLEVBQUUsT0FBTyxDQUFDLEVBQUUsVUFBVSxDQUFhLENBQUM7SUFDNUUsTUFBTSxpQkFBaUIsR0FDbkIsTUFBTSxDQUFDLElBQUksQ0FBQyxZQUFZLEVBQUUsT0FBTyxDQUFDLEVBQUUsVUFBVSxDQUFhLENBQUM7SUFDaEUsTUFBTSxhQUFhLEdBQWEsU0FBUyxDQUFDLFlBQVksQ0FBQyxDQUFDO0lBQ3hELE1BQU0sT0FBTyxHQUFhLE1BQU0sQ0FBQyxhQUFhLEVBQUUsaUJBQWlCLENBQUMsQ0FBQztJQUNuRSxPQUFPLElBQUksQ0FBQyxPQUFPLEVBQUUsT0FBTyxDQUFDLENBQUM7QUFDaEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGVBQWUsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsZ0JBQWdCLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvcjFELCBUZW5zb3IyRH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge2Nhc3R9IGZyb20gJy4vY2FzdCc7XG5pbXBvcnQge21hdE11bH0gZnJvbSAnLi9tYXRfbXVsJztcbmltcG9ydCB7b25lSG90fSBmcm9tICcuL29uZV9ob3QnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuaW1wb3J0IHt0cmFuc3Bvc2V9IGZyb20gJy4vdHJhbnNwb3NlJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgY29uZnVzaW9uIG1hdHJpeCBmcm9tIHRydWUgbGFiZWxzIGFuZCBwcmVkaWN0ZWQgbGFiZWxzLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBsYWJlbHMgPSB0Zi50ZW5zb3IxZChbMCwgMSwgMiwgMSwgMF0sICdpbnQzMicpO1xuICogY29uc3QgcHJlZGljdGlvbnMgPSB0Zi50ZW5zb3IxZChbMCwgMiwgMiwgMSwgMF0sICdpbnQzMicpO1xuICogY29uc3QgbnVtQ2xhc3NlcyA9IDM7XG4gKiBjb25zdCBvdXQgPSB0Zi5tYXRoLmNvbmZ1c2lvbk1hdHJpeChsYWJlbHMsIHByZWRpY3Rpb25zLCBudW1DbGFzc2VzKTtcbiAqIG91dC5wcmludCgpO1xuICogLy8gRXhwZWN0ZWQgb3V0cHV0IG1hdHJpeDpcbiAqIC8vIFtbMiwgMCwgMF0sXG4gKiAvLyAgWzAsIDEsIDFdLFxuICogLy8gIFswLCAwLCAxXV1cbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBsYWJlbHMgVGhlIHRhcmdldCBsYWJlbHMsIGFzc3VtZWQgdG8gYmUgMC1iYXNlZCBpbnRlZ2Vyc1xuICogICBmb3IgdGhlIGNsYXNzZXMuIFRoZSBzaGFwZSBpcyBgW251bUV4YW1wbGVzXWAsIHdoZXJlXG4gKiAgIGBudW1FeGFtcGxlc2AgaXMgdGhlIG51bWJlciBvZiBleGFtcGxlcyBpbmNsdWRlZC5cbiAqIEBwYXJhbSBwcmVkaWN0aW9ucyBUaGUgcHJlZGljdGVkIGNsYXNzZXMsIGFzc3VtZWQgdG8gYmVcbiAqICAgMC1iYXNlZCBpbnRlZ2VycyBmb3IgdGhlIGNsYXNzZXMuIE11c3QgaGF2ZSB0aGUgc2FtZSBzaGFwZSBhcyBgbGFiZWxzYC5cbiAqIEBwYXJhbSBudW1DbGFzc2VzIE51bWJlciBvZiBhbGwgY2xhc3NlcywgYXMgYW4gaW50ZWdlci5cbiAqICAgSXRzIHZhbHVlIG11c3QgYmUgbGFyZ2VyIHRoYW4gdGhlIGxhcmdlc3QgZWxlbWVudCBpbiBgbGFiZWxzYCBhbmRcbiAqICAgYHByZWRpY3Rpb25zYC5cbiAqIEByZXR1cm5zIFRoZSBjb25mdXNpb24gbWF0cml4IGFzIGEgaW50MzItdHlwZSAyRCB0ZW5zb3IuIFRoZSB2YWx1ZSBhdFxuICogICByb3cgYHJgIGFuZCBjb2x1bW4gYGNgIGlzIHRoZSBudW1iZXIgb2YgdGltZXMgZXhhbXBsZXMgb2YgYWN0dWFsIGNsYXNzXG4gKiAgIGByYCB3ZXJlIHByZWRpY3RlZCBhcyBjbGFzcyBgY2AuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnRXZhbHVhdGlvbid9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjb25mdXNpb25NYXRyaXhfKFxuICAgIGxhYmVsczogVGVuc29yMUR8VGVuc29yTGlrZSwgcHJlZGljdGlvbnM6IFRlbnNvcjFEfFRlbnNvckxpa2UsXG4gICAgbnVtQ2xhc3NlczogbnVtYmVyKTogVGVuc29yMkQge1xuICBjb25zdCAkbGFiZWxzID0gY29udmVydFRvVGVuc29yKGxhYmVscywgJ2xhYmVscycsICdjb25mdXNpb25NYXRyaXgnKTtcbiAgY29uc3QgJHByZWRpY3Rpb25zID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihwcmVkaWN0aW9ucywgJ3ByZWRpY3Rpb25zJywgJ2NvbmZ1c2lvbk1hdHJpeCcpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgbnVtQ2xhc3NlcyA9PSBudWxsIHx8IG51bUNsYXNzZXMgPiAwICYmIE51bWJlci5pc0ludGVnZXIobnVtQ2xhc3NlcyksXG4gICAgICAoKSA9PiBgSWYgcHJvdmlkZWQsIG51bUNsYXNzZXMgbXVzdCBiZSBhIHBvc2l0aXZlIGludGVnZXIsIGAgK1xuICAgICAgICAgIGBidXQgZ290ICR7bnVtQ2xhc3Nlc31gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkbGFiZWxzLnJhbmsgPT09IDEsXG4gICAgICAoKSA9PiBgRXhwZWN0ZWQgdGhlIHJhbmsgb2YgbGFiZWxzIHRvIGJlIDEsIGJ1dCBnb3QgJHskbGFiZWxzLnJhbmt9YCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHByZWRpY3Rpb25zLnJhbmsgPT09IDEsXG4gICAgICAoKSA9PiBgRXhwZWN0ZWQgdGhlIHJhbmsgb2YgcHJlZGljdGlvbnMgdG8gYmUgMSwgYCArXG4gICAgICAgICAgYGJ1dCBnb3QgJHskcHJlZGljdGlvbnMucmFua31gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkbGFiZWxzLnNoYXBlWzBdID09PSAkcHJlZGljdGlvbnMuc2hhcGVbMF0sXG4gICAgICAoKSA9PiBgTWlzbWF0Y2ggaW4gdGhlIG51bWJlciBvZiBleGFtcGxlczogYCArXG4gICAgICAgICAgYCR7JGxhYmVscy5zaGFwZVswXX0gdnMuICR7JHByZWRpY3Rpb25zLnNoYXBlWzBdfS4gYCArXG4gICAgICAgICAgYExhYmVscyBhbmQgcHJlZGljdGlvbnMgc2hvdWxkIGhhdmUgdGhlIHNhbWUgbnVtYmVyIG9mIGVsZW1lbnRzLmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIG51bUNsYXNzZXMgPiAwICYmIE51bWJlci5pc0ludGVnZXIobnVtQ2xhc3NlcyksXG4gICAgICAoKSA9PiBgbnVtQ2xhc3NlcyBpcyByZXF1aXJlZCB0byBiZSBhIHBvc2l0aXZlIGludGVnZXIsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgYCR7bnVtQ2xhc3Nlc31gKTtcbiAgLy8gVE9ETyhjYWlzKTogSW4gdGhlIGZ1dHVyZSwgaWYgb25lSG90IHN1cHBvcnRzIHRlbnNvcnMgaW5wdXRzIGZvclxuICAvLyAgIGBudW1DbGFzc2VzYCwgYGNvbmZ1c2lvbk1hdHJpeGAgY2FuIG1ha2UgYG51bUNsYXNzZXNgIG9wdGlvbmFsLlxuXG4gIGNvbnN0IG9uZUhvdExhYmVscyA9IG9uZUhvdChjYXN0KCRsYWJlbHMsICdpbnQzMicpLCBudW1DbGFzc2VzKSBhcyBUZW5zb3IyRDtcbiAgY29uc3Qgb25lSG90UHJlZGljdGlvbnMgPVxuICAgICAgb25lSG90KGNhc3QoJHByZWRpY3Rpb25zLCAnaW50MzInKSwgbnVtQ2xhc3NlcykgYXMgVGVuc29yMkQ7XG4gIGNvbnN0IG9uZUhvdExhYmVsc1Q6IFRlbnNvcjJEID0gdHJhbnNwb3NlKG9uZUhvdExhYmVscyk7XG4gIGNvbnN0IHByb2R1Y3Q6IFRlbnNvcjJEID0gbWF0TXVsKG9uZUhvdExhYmVsc1QsIG9uZUhvdFByZWRpY3Rpb25zKTtcbiAgcmV0dXJuIGNhc3QocHJvZHVjdCwgJ2ludDMyJyk7XG59XG5cbmV4cG9ydCBjb25zdCBjb25mdXNpb25NYXRyaXggPSAvKiBAX19QVVJFX18gKi8gb3Aoe2NvbmZ1c2lvbk1hdHJpeF99KTtcbiJdfQ==