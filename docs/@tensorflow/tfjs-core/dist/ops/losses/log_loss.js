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
import { add } from '../add';
import { log } from '../log';
import { Reduction } from '../loss_ops_utils';
import { mul } from '../mul';
import { neg } from '../neg';
import { op } from '../operation';
import { scalar } from '../scalar';
import { sub } from '../sub';
import { computeWeightedLoss } from './compute_weighted_loss';
/**
 * Computes the log loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param epsilon A small increment to avoid taking log of zero
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function logLoss_(labels, predictions, weights, epsilon = 1e-7, reduction = Reduction.SUM_BY_NONZERO_WEIGHTS) {
    const $labels = convertToTensor(labels, 'labels', 'logLoss');
    const $predictions = convertToTensor(predictions, 'predictions', 'logLoss');
    let $weights = null;
    if (weights != null) {
        $weights = convertToTensor(weights, 'weights', 'logLoss');
    }
    assertShapesMatch($labels.shape, $predictions.shape, 'Error in logLoss: ');
    const one = scalar(1);
    const epsilonScalar = scalar(epsilon);
    const l1 = neg(mul($labels, log(add($predictions, epsilonScalar))));
    const l2 = mul(sub(one, $labels), log(add(sub(one, $predictions), epsilonScalar)));
    const losses = sub(l1, l2);
    return computeWeightedLoss(losses, $weights, reduction);
}
export const logLoss = /* @__PURE__ */ op({ logLoss_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibG9nX2xvc3MuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9sb3NzZXMvbG9nX2xvc3MudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXRELE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUM3QyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDM0IsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQzVDLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDM0IsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUMzQixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ2hDLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUUzQixPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUU1RDs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxTQUFTLFFBQVEsQ0FDYixNQUFvQixFQUFFLFdBQXlCLEVBQy9DLE9BQTJCLEVBQUUsT0FBTyxHQUFHLElBQUksRUFDM0MsU0FBUyxHQUFHLFNBQVMsQ0FBQyxzQkFBc0I7SUFDOUMsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDN0QsTUFBTSxZQUFZLEdBQUcsZUFBZSxDQUFDLFdBQVcsRUFBRSxhQUFhLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDNUUsSUFBSSxRQUFRLEdBQVcsSUFBSSxDQUFDO0lBQzVCLElBQUksT0FBTyxJQUFJLElBQUksRUFBRTtRQUNuQixRQUFRLEdBQUcsZUFBZSxDQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUsU0FBUyxDQUFDLENBQUM7S0FDM0Q7SUFDRCxpQkFBaUIsQ0FBQyxPQUFPLENBQUMsS0FBSyxFQUFFLFlBQVksQ0FBQyxLQUFLLEVBQUUsb0JBQW9CLENBQUMsQ0FBQztJQUUzRSxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDdEIsTUFBTSxhQUFhLEdBQUcsTUFBTSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBRXRDLE1BQU0sRUFBRSxHQUFHLEdBQUcsQ0FBQyxHQUFHLENBQUMsT0FBTyxFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsWUFBWSxFQUFFLGFBQWEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3BFLE1BQU0sRUFBRSxHQUNKLEdBQUcsQ0FBQyxHQUFHLENBQUMsR0FBRyxFQUFFLE9BQU8sQ0FBQyxFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLEdBQUcsRUFBRSxZQUFZLENBQUMsRUFBRSxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDNUUsTUFBTSxNQUFNLEdBQUcsR0FBRyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQztJQUMzQixPQUFPLG1CQUFtQixDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsU0FBUyxDQUFDLENBQUM7QUFDMUQsQ0FBQztBQUNELE1BQU0sQ0FBQyxNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsUUFBUSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0IHthc3NlcnRTaGFwZXNNYXRjaH0gZnJvbSAnLi4vLi4vdXRpbCc7XG5pbXBvcnQge2FkZH0gZnJvbSAnLi4vYWRkJztcbmltcG9ydCB7bG9nfSBmcm9tICcuLi9sb2cnO1xuaW1wb3J0IHtSZWR1Y3Rpb259IGZyb20gJy4uL2xvc3Nfb3BzX3V0aWxzJztcbmltcG9ydCB7bXVsfSBmcm9tICcuLi9tdWwnO1xuaW1wb3J0IHtuZWd9IGZyb20gJy4uL25lZyc7XG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtzY2FsYXJ9IGZyb20gJy4uL3NjYWxhcic7XG5pbXBvcnQge3N1Yn0gZnJvbSAnLi4vc3ViJztcblxuaW1wb3J0IHtjb21wdXRlV2VpZ2h0ZWRMb3NzfSBmcm9tICcuL2NvbXB1dGVfd2VpZ2h0ZWRfbG9zcyc7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIGxvZyBsb3NzIGJldHdlZW4gdHdvIHRlbnNvcnMuXG4gKlxuICogQHBhcmFtIGxhYmVscyBUaGUgZ3JvdW5kIHRydXRoIG91dHB1dCB0ZW5zb3IsIHNhbWUgZGltZW5zaW9ucyBhc1xuICogICAgJ3ByZWRpY3Rpb25zJy5cbiAqIEBwYXJhbSBwcmVkaWN0aW9ucyBUaGUgcHJlZGljdGVkIG91dHB1dHMuXG4gKiBAcGFyYW0gd2VpZ2h0cyBUZW5zb3Igd2hvc2UgcmFuayBpcyBlaXRoZXIgMCwgb3IgdGhlIHNhbWUgcmFuayBhc1xuICogICAgYGxhYmVsc2AsIGFuZCBtdXN0IGJlIGJyb2FkY2FzdGFibGUgdG8gYGxhYmVsc2AgKGkuZS4sIGFsbCBkaW1lbnNpb25zXG4gKiAgICBtdXN0IGJlIGVpdGhlciBgMWAsIG9yIHRoZSBzYW1lIGFzIHRoZSBjb3JyZXNwb25kaW5nIGBsb3NzZXNgXG4gKiAgICBkaW1lbnNpb24pLlxuICogQHBhcmFtIGVwc2lsb24gQSBzbWFsbCBpbmNyZW1lbnQgdG8gYXZvaWQgdGFraW5nIGxvZyBvZiB6ZXJvXG4gKiBAcGFyYW0gcmVkdWN0aW9uIFR5cGUgb2YgcmVkdWN0aW9uIHRvIGFwcGx5IHRvIGxvc3MuIFNob3VsZCBiZSBvZiB0eXBlXG4gKiAgICBgUmVkdWN0aW9uYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUcmFpbmluZycsIHN1YmhlYWRpbmc6ICdMb3NzZXMnLCBuYW1lc3BhY2U6ICdsb3NzZXMnfVxuICovXG5mdW5jdGlvbiBsb2dMb3NzXzxUIGV4dGVuZHMgVGVuc29yLCBPIGV4dGVuZHMgVGVuc29yPihcbiAgICBsYWJlbHM6IFR8VGVuc29yTGlrZSwgcHJlZGljdGlvbnM6IFR8VGVuc29yTGlrZSxcbiAgICB3ZWlnaHRzPzogVGVuc29yfFRlbnNvckxpa2UsIGVwc2lsb24gPSAxZS03LFxuICAgIHJlZHVjdGlvbiA9IFJlZHVjdGlvbi5TVU1fQllfTk9OWkVST19XRUlHSFRTKTogTyB7XG4gIGNvbnN0ICRsYWJlbHMgPSBjb252ZXJ0VG9UZW5zb3IobGFiZWxzLCAnbGFiZWxzJywgJ2xvZ0xvc3MnKTtcbiAgY29uc3QgJHByZWRpY3Rpb25zID0gY29udmVydFRvVGVuc29yKHByZWRpY3Rpb25zLCAncHJlZGljdGlvbnMnLCAnbG9nTG9zcycpO1xuICBsZXQgJHdlaWdodHM6IFRlbnNvciA9IG51bGw7XG4gIGlmICh3ZWlnaHRzICE9IG51bGwpIHtcbiAgICAkd2VpZ2h0cyA9IGNvbnZlcnRUb1RlbnNvcih3ZWlnaHRzLCAnd2VpZ2h0cycsICdsb2dMb3NzJyk7XG4gIH1cbiAgYXNzZXJ0U2hhcGVzTWF0Y2goJGxhYmVscy5zaGFwZSwgJHByZWRpY3Rpb25zLnNoYXBlLCAnRXJyb3IgaW4gbG9nTG9zczogJyk7XG5cbiAgY29uc3Qgb25lID0gc2NhbGFyKDEpO1xuICBjb25zdCBlcHNpbG9uU2NhbGFyID0gc2NhbGFyKGVwc2lsb24pO1xuXG4gIGNvbnN0IGwxID0gbmVnKG11bCgkbGFiZWxzLCBsb2coYWRkKCRwcmVkaWN0aW9ucywgZXBzaWxvblNjYWxhcikpKSk7XG4gIGNvbnN0IGwyID1cbiAgICAgIG11bChzdWIob25lLCAkbGFiZWxzKSwgbG9nKGFkZChzdWIob25lLCAkcHJlZGljdGlvbnMpLCBlcHNpbG9uU2NhbGFyKSkpO1xuICBjb25zdCBsb3NzZXMgPSBzdWIobDEsIGwyKTtcbiAgcmV0dXJuIGNvbXB1dGVXZWlnaHRlZExvc3MobG9zc2VzLCAkd2VpZ2h0cywgcmVkdWN0aW9uKTtcbn1cbmV4cG9ydCBjb25zdCBsb2dMb3NzID0gLyogQF9fUFVSRV9fICovIG9wKHtsb2dMb3NzX30pO1xuIl19