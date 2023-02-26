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
import { convertToTensor } from '../../tensor_util_env';
import { assertShapesMatch } from '../../util';
import { abs } from '../abs';
import { add } from '../add';
import { Reduction } from '../loss_ops_utils';
import { minimum } from '../minimum';
import { mul } from '../mul';
import { op } from '../operation';
import { scalar } from '../scalar';
import { square } from '../square';
import { sub } from '../sub';
import { computeWeightedLoss } from './compute_weighted_loss';
/**
 * Computes the Huber loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param delta Point where Huber loss changes from quadratic to linear.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`.
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function huberLoss_(labels, predictions, weights, delta = 1.0, reduction = Reduction.SUM_BY_NONZERO_WEIGHTS) {
    const $labels = convertToTensor(labels, 'labels', 'huberLoss');
    const $predictions = convertToTensor(predictions, 'predictions', 'huberLoss');
    let $weights = null;
    if (weights != null) {
        $weights = convertToTensor(weights, 'weights', 'huberLoss');
    }
    assertShapesMatch($labels.shape, $predictions.shape, 'Error in huberLoss: ');
    const deltaScalar = scalar(delta);
    const error = abs(sub($predictions, $labels));
    const quadratic = minimum(error, deltaScalar);
    const linear = sub(error, quadratic);
    const losses = add(mul(scalar(0.5), square(quadratic)), mul(deltaScalar, linear));
    return computeWeightedLoss(losses, $weights, reduction);
}
export const huberLoss = /* @__PURE__ */ op({ huberLoss_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaHViZXJfbG9zcy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2xvc3Nlcy9odWJlcl9sb3NzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUdILE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV0RCxPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDN0MsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUMzQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxTQUFTLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUM1QyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQ25DLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDM0IsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUUzQixPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUU1RDs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxTQUFTLFVBQVUsQ0FDZixNQUFvQixFQUFFLFdBQXlCLEVBQy9DLE9BQTJCLEVBQUUsS0FBSyxHQUFHLEdBQUcsRUFDeEMsU0FBUyxHQUFHLFNBQVMsQ0FBQyxzQkFBc0I7SUFDOUMsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFDL0QsTUFBTSxZQUFZLEdBQUcsZUFBZSxDQUFDLFdBQVcsRUFBRSxhQUFhLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFDOUUsSUFBSSxRQUFRLEdBQVcsSUFBSSxDQUFDO0lBQzVCLElBQUksT0FBTyxJQUFJLElBQUksRUFBRTtRQUNuQixRQUFRLEdBQUcsZUFBZSxDQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUsV0FBVyxDQUFDLENBQUM7S0FDN0Q7SUFDRCxpQkFBaUIsQ0FBQyxPQUFPLENBQUMsS0FBSyxFQUFFLFlBQVksQ0FBQyxLQUFLLEVBQUUsc0JBQXNCLENBQUMsQ0FBQztJQUU3RSxNQUFNLFdBQVcsR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDbEMsTUFBTSxLQUFLLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQyxZQUFZLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUM5QyxNQUFNLFNBQVMsR0FBRyxPQUFPLENBQUMsS0FBSyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBQzlDLE1BQU0sTUFBTSxHQUFHLEdBQUcsQ0FBQyxLQUFLLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFckMsTUFBTSxNQUFNLEdBQ1IsR0FBRyxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFDLEVBQUUsTUFBTSxDQUFDLFNBQVMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLFdBQVcsRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDO0lBQ3ZFLE9BQU8sbUJBQW1CLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxTQUFTLENBQUMsQ0FBQztBQUMxRCxDQUFDO0FBQ0QsTUFBTSxDQUFDLE1BQU0sU0FBUyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxVQUFVLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi8uLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi8uLi90eXBlcyc7XG5pbXBvcnQge2Fzc2VydFNoYXBlc01hdGNofSBmcm9tICcuLi8uLi91dGlsJztcbmltcG9ydCB7YWJzfSBmcm9tICcuLi9hYnMnO1xuaW1wb3J0IHthZGR9IGZyb20gJy4uL2FkZCc7XG5pbXBvcnQge1JlZHVjdGlvbn0gZnJvbSAnLi4vbG9zc19vcHNfdXRpbHMnO1xuaW1wb3J0IHttaW5pbXVtfSBmcm9tICcuLi9taW5pbXVtJztcbmltcG9ydCB7bXVsfSBmcm9tICcuLi9tdWwnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi4vb3BlcmF0aW9uJztcbmltcG9ydCB7c2NhbGFyfSBmcm9tICcuLi9zY2FsYXInO1xuaW1wb3J0IHtzcXVhcmV9IGZyb20gJy4uL3NxdWFyZSc7XG5pbXBvcnQge3N1Yn0gZnJvbSAnLi4vc3ViJztcblxuaW1wb3J0IHtjb21wdXRlV2VpZ2h0ZWRMb3NzfSBmcm9tICcuL2NvbXB1dGVfd2VpZ2h0ZWRfbG9zcyc7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIEh1YmVyIGxvc3MgYmV0d2VlbiB0d28gdGVuc29ycy5cbiAqXG4gKiBAcGFyYW0gbGFiZWxzIFRoZSBncm91bmQgdHJ1dGggb3V0cHV0IHRlbnNvciwgc2FtZSBkaW1lbnNpb25zIGFzXG4gKiAgICAncHJlZGljdGlvbnMnLlxuICogQHBhcmFtIHByZWRpY3Rpb25zIFRoZSBwcmVkaWN0ZWQgb3V0cHV0cy5cbiAqIEBwYXJhbSB3ZWlnaHRzIFRlbnNvciB3aG9zZSByYW5rIGlzIGVpdGhlciAwLCBvciB0aGUgc2FtZSByYW5rIGFzXG4gKiAgICBgbGFiZWxzYCwgYW5kIG11c3QgYmUgYnJvYWRjYXN0YWJsZSB0byBgbGFiZWxzYCAoaS5lLiwgYWxsIGRpbWVuc2lvbnNcbiAqICAgIG11c3QgYmUgZWl0aGVyIGAxYCwgb3IgdGhlIHNhbWUgYXMgdGhlIGNvcnJlc3BvbmRpbmcgYGxvc3Nlc2BcbiAqICAgIGRpbWVuc2lvbikuXG4gKiBAcGFyYW0gZGVsdGEgUG9pbnQgd2hlcmUgSHViZXIgbG9zcyBjaGFuZ2VzIGZyb20gcXVhZHJhdGljIHRvIGxpbmVhci5cbiAqIEBwYXJhbSByZWR1Y3Rpb24gVHlwZSBvZiByZWR1Y3Rpb24gdG8gYXBwbHkgdG8gbG9zcy4gU2hvdWxkIGJlIG9mIHR5cGVcbiAqICAgIGBSZWR1Y3Rpb25gLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUcmFpbmluZycsIHN1YmhlYWRpbmc6ICdMb3NzZXMnLCBuYW1lc3BhY2U6ICdsb3NzZXMnfVxuICovXG5mdW5jdGlvbiBodWJlckxvc3NfPFQgZXh0ZW5kcyBUZW5zb3IsIE8gZXh0ZW5kcyBUZW5zb3I+KFxuICAgIGxhYmVsczogVHxUZW5zb3JMaWtlLCBwcmVkaWN0aW9uczogVHxUZW5zb3JMaWtlLFxuICAgIHdlaWdodHM/OiBUZW5zb3J8VGVuc29yTGlrZSwgZGVsdGEgPSAxLjAsXG4gICAgcmVkdWN0aW9uID0gUmVkdWN0aW9uLlNVTV9CWV9OT05aRVJPX1dFSUdIVFMpOiBPIHtcbiAgY29uc3QgJGxhYmVscyA9IGNvbnZlcnRUb1RlbnNvcihsYWJlbHMsICdsYWJlbHMnLCAnaHViZXJMb3NzJyk7XG4gIGNvbnN0ICRwcmVkaWN0aW9ucyA9IGNvbnZlcnRUb1RlbnNvcihwcmVkaWN0aW9ucywgJ3ByZWRpY3Rpb25zJywgJ2h1YmVyTG9zcycpO1xuICBsZXQgJHdlaWdodHM6IFRlbnNvciA9IG51bGw7XG4gIGlmICh3ZWlnaHRzICE9IG51bGwpIHtcbiAgICAkd2VpZ2h0cyA9IGNvbnZlcnRUb1RlbnNvcih3ZWlnaHRzLCAnd2VpZ2h0cycsICdodWJlckxvc3MnKTtcbiAgfVxuICBhc3NlcnRTaGFwZXNNYXRjaCgkbGFiZWxzLnNoYXBlLCAkcHJlZGljdGlvbnMuc2hhcGUsICdFcnJvciBpbiBodWJlckxvc3M6ICcpO1xuXG4gIGNvbnN0IGRlbHRhU2NhbGFyID0gc2NhbGFyKGRlbHRhKTtcbiAgY29uc3QgZXJyb3IgPSBhYnMoc3ViKCRwcmVkaWN0aW9ucywgJGxhYmVscykpO1xuICBjb25zdCBxdWFkcmF0aWMgPSBtaW5pbXVtKGVycm9yLCBkZWx0YVNjYWxhcik7XG4gIGNvbnN0IGxpbmVhciA9IHN1YihlcnJvciwgcXVhZHJhdGljKTtcblxuICBjb25zdCBsb3NzZXMgPVxuICAgICAgYWRkKG11bChzY2FsYXIoMC41KSwgc3F1YXJlKHF1YWRyYXRpYykpLCBtdWwoZGVsdGFTY2FsYXIsIGxpbmVhcikpO1xuICByZXR1cm4gY29tcHV0ZVdlaWdodGVkTG9zcyhsb3NzZXMsICR3ZWlnaHRzLCByZWR1Y3Rpb24pO1xufVxuZXhwb3J0IGNvbnN0IGh1YmVyTG9zcyA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7aHViZXJMb3NzX30pO1xuIl19