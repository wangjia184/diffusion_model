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
import { exp } from '../exp';
import { log1p } from '../log1p';
import { Reduction } from '../loss_ops_utils';
import { mul } from '../mul';
import { neg } from '../neg';
import { op } from '../operation';
import { relu } from '../relu';
import { scalar } from '../scalar';
import { sub } from '../sub';
import { computeWeightedLoss } from './compute_weighted_loss';
function sigmoidCrossEntropyWithLogits_(labels, logits) {
    const $labels = convertToTensor(labels, 'labels', 'sigmoidCrossEntropyWithLogits');
    const $logits = convertToTensor(logits, 'logits', 'sigmoidCrossEntropyWithLogits');
    assertShapesMatch($labels.shape, $logits.shape, 'Error in sigmoidCrossEntropyWithLogits: ');
    /**
     * Implementation Details:
     *
     * For brevity, let `x = logits`, `z = labels`.  The logistic loss is
     *     z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
     *   = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
     *   = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
     *   = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
     *   = (1 - z) * x + log(1 + exp(-x))
     *   = x - x * z + log(1 + exp(-x))
     *
     *   For x < 0, to avoid overflow in exp(-x), we reformulate the above
     *     x - x * z + log(1 + exp(-x))
     *   = log(exp(x)) - x * z + log(1 + exp(-x))
     *   = - x * z + log(1 + exp(x))
     *
     * Hence, to ensure stability and avoid overflow, the implementation uses
     * this equivalent formulation:
     *     max(x, 0) - x * z + log(1 + exp(-abs(x)))
     */
    const maxOutput = relu($logits);
    const outputXTarget = mul($logits, $labels);
    const sigmoidOutput = log1p(exp(neg(abs($logits))));
    return add(sub(maxOutput, outputXTarget), sigmoidOutput);
}
/**
 * Computes the sigmoid cross entropy loss between two tensors.
 *
 * If labelSmoothing is nonzero, smooth the labels towards 1/2:
 *
 *   newMulticlassLabels = multiclassLabels * (1 - labelSmoothing)
 *                         + 0.5 * labelSmoothing
 *
 * @param multiClassLabels The ground truth output tensor of shape
 * [batch_size, num_classes], same dimensions as 'predictions'.
 * @param logits The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param labelSmoothing If greater than 0, then smooth the labels.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' }
 */
function sigmoidCrossEntropy_(multiClassLabels, logits, weights, labelSmoothing = 0, reduction = Reduction.SUM_BY_NONZERO_WEIGHTS) {
    let $multiClassLabels = convertToTensor(multiClassLabels, 'multiClassLabels', 'sigmoidCrossEntropy');
    const $logits = convertToTensor(logits, 'logits', 'sigmoidCrossEntropy');
    let $weights = null;
    if (weights != null) {
        $weights = convertToTensor(weights, 'weights', 'sigmoidCrossEntropy');
    }
    assertShapesMatch($multiClassLabels.shape, $logits.shape, 'Error in sigmoidCrossEntropy: ');
    if (labelSmoothing > 0) {
        const labelSmoothingScalar = scalar(labelSmoothing);
        const one = scalar(1);
        const half = scalar(0.5);
        $multiClassLabels =
            add(mul($multiClassLabels, sub(one, labelSmoothingScalar)), mul(half, labelSmoothingScalar));
    }
    const losses = sigmoidCrossEntropyWithLogits_($multiClassLabels, $logits);
    return computeWeightedLoss(losses, $weights, reduction);
}
export const sigmoidCrossEntropy = /* @__PURE__ */ op({ sigmoidCrossEntropy_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2lnbW9pZF9jcm9zc19lbnRyb3B5LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvbG9zc2VzL3NpZ21vaWRfY3Jvc3NfZW50cm9weS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxFQUFDLGlCQUFpQixFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQzdDLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDM0IsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUMzQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDL0IsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQzVDLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDM0IsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUMzQixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ2hDLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDN0IsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBRTNCLE9BQU8sRUFBQyxtQkFBbUIsRUFBQyxNQUFNLHlCQUF5QixDQUFDO0FBRTVELFNBQVMsOEJBQThCLENBQ25DLE1BQW9CLEVBQUUsTUFBb0I7SUFDNUMsTUFBTSxPQUFPLEdBQ1QsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsK0JBQStCLENBQUMsQ0FBQztJQUN2RSxNQUFNLE9BQU8sR0FDVCxlQUFlLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSwrQkFBK0IsQ0FBQyxDQUFDO0lBQ3ZFLGlCQUFpQixDQUNiLE9BQU8sQ0FBQyxLQUFLLEVBQUUsT0FBTyxDQUFDLEtBQUssRUFBRSwwQ0FBMEMsQ0FBQyxDQUFDO0lBRTlFOzs7Ozs7Ozs7Ozs7Ozs7Ozs7O09BbUJHO0lBQ0gsTUFBTSxTQUFTLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQ2hDLE1BQU0sYUFBYSxHQUFHLEdBQUcsQ0FBQyxPQUFPLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDNUMsTUFBTSxhQUFhLEdBQUcsS0FBSyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRXBELE9BQU8sR0FBRyxDQUFDLEdBQUcsQ0FBQyxTQUFTLEVBQUUsYUFBYSxDQUFDLEVBQUUsYUFBYSxDQUFDLENBQUM7QUFDM0QsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQW9CRztBQUNILFNBQVMsb0JBQW9CLENBQ3pCLGdCQUE4QixFQUFFLE1BQW9CLEVBQ3BELE9BQTJCLEVBQUUsY0FBYyxHQUFHLENBQUMsRUFDL0MsU0FBUyxHQUFHLFNBQVMsQ0FBQyxzQkFBc0I7SUFDOUMsSUFBSSxpQkFBaUIsR0FBRyxlQUFlLENBQ25DLGdCQUFnQixFQUFFLGtCQUFrQixFQUFFLHFCQUFxQixDQUFDLENBQUM7SUFDakUsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUscUJBQXFCLENBQUMsQ0FBQztJQUN6RSxJQUFJLFFBQVEsR0FBVyxJQUFJLENBQUM7SUFDNUIsSUFBSSxPQUFPLElBQUksSUFBSSxFQUFFO1FBQ25CLFFBQVEsR0FBRyxlQUFlLENBQUMsT0FBTyxFQUFFLFNBQVMsRUFBRSxxQkFBcUIsQ0FBQyxDQUFDO0tBQ3ZFO0lBQ0QsaUJBQWlCLENBQ2IsaUJBQWlCLENBQUMsS0FBSyxFQUFFLE9BQU8sQ0FBQyxLQUFLLEVBQUUsZ0NBQWdDLENBQUMsQ0FBQztJQUU5RSxJQUFJLGNBQWMsR0FBRyxDQUFDLEVBQUU7UUFDdEIsTUFBTSxvQkFBb0IsR0FBRyxNQUFNLENBQUMsY0FBYyxDQUFDLENBQUM7UUFDcEQsTUFBTSxHQUFHLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3RCLE1BQU0sSUFBSSxHQUFHLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUV6QixpQkFBaUI7WUFDYixHQUFHLENBQUMsR0FBRyxDQUFDLGlCQUFpQixFQUFFLEdBQUcsQ0FBQyxHQUFHLEVBQUUsb0JBQW9CLENBQUMsQ0FBQyxFQUN0RCxHQUFHLENBQUMsSUFBSSxFQUFFLG9CQUFvQixDQUFDLENBQUMsQ0FBQztLQUMxQztJQUNELE1BQU0sTUFBTSxHQUFHLDhCQUE4QixDQUFDLGlCQUFpQixFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBRTFFLE9BQU8sbUJBQW1CLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxTQUFTLENBQUMsQ0FBQztBQUMxRCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sbUJBQW1CLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLG9CQUFvQixFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0IHthc3NlcnRTaGFwZXNNYXRjaH0gZnJvbSAnLi4vLi4vdXRpbCc7XG5pbXBvcnQge2Fic30gZnJvbSAnLi4vYWJzJztcbmltcG9ydCB7YWRkfSBmcm9tICcuLi9hZGQnO1xuaW1wb3J0IHtleHB9IGZyb20gJy4uL2V4cCc7XG5pbXBvcnQge2xvZzFwfSBmcm9tICcuLi9sb2cxcCc7XG5pbXBvcnQge1JlZHVjdGlvbn0gZnJvbSAnLi4vbG9zc19vcHNfdXRpbHMnO1xuaW1wb3J0IHttdWx9IGZyb20gJy4uL211bCc7XG5pbXBvcnQge25lZ30gZnJvbSAnLi4vbmVnJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5pbXBvcnQge3JlbHV9IGZyb20gJy4uL3JlbHUnO1xuaW1wb3J0IHtzY2FsYXJ9IGZyb20gJy4uL3NjYWxhcic7XG5pbXBvcnQge3N1Yn0gZnJvbSAnLi4vc3ViJztcblxuaW1wb3J0IHtjb21wdXRlV2VpZ2h0ZWRMb3NzfSBmcm9tICcuL2NvbXB1dGVfd2VpZ2h0ZWRfbG9zcyc7XG5cbmZ1bmN0aW9uIHNpZ21vaWRDcm9zc0VudHJvcHlXaXRoTG9naXRzXzxUIGV4dGVuZHMgVGVuc29yLCBPIGV4dGVuZHMgVGVuc29yPihcbiAgICBsYWJlbHM6IFR8VGVuc29yTGlrZSwgbG9naXRzOiBUfFRlbnNvckxpa2UpOiBPIHtcbiAgY29uc3QgJGxhYmVscyA9XG4gICAgICBjb252ZXJ0VG9UZW5zb3IobGFiZWxzLCAnbGFiZWxzJywgJ3NpZ21vaWRDcm9zc0VudHJvcHlXaXRoTG9naXRzJyk7XG4gIGNvbnN0ICRsb2dpdHMgPVxuICAgICAgY29udmVydFRvVGVuc29yKGxvZ2l0cywgJ2xvZ2l0cycsICdzaWdtb2lkQ3Jvc3NFbnRyb3B5V2l0aExvZ2l0cycpO1xuICBhc3NlcnRTaGFwZXNNYXRjaChcbiAgICAgICRsYWJlbHMuc2hhcGUsICRsb2dpdHMuc2hhcGUsICdFcnJvciBpbiBzaWdtb2lkQ3Jvc3NFbnRyb3B5V2l0aExvZ2l0czogJyk7XG5cbiAgLyoqXG4gICAqIEltcGxlbWVudGF0aW9uIERldGFpbHM6XG4gICAqXG4gICAqIEZvciBicmV2aXR5LCBsZXQgYHggPSBsb2dpdHNgLCBgeiA9IGxhYmVsc2AuICBUaGUgbG9naXN0aWMgbG9zcyBpc1xuICAgKiAgICAgeiAqIC1sb2coc2lnbW9pZCh4KSkgKyAoMSAtIHopICogLWxvZygxIC0gc2lnbW9pZCh4KSlcbiAgICogICA9IHogKiAtbG9nKDEgLyAoMSArIGV4cCgteCkpKSArICgxIC0geikgKiAtbG9nKGV4cCgteCkgLyAoMSArIGV4cCgteCkpKVxuICAgKiAgID0geiAqIGxvZygxICsgZXhwKC14KSkgKyAoMSAtIHopICogKC1sb2coZXhwKC14KSkgKyBsb2coMSArIGV4cCgteCkpKVxuICAgKiAgID0geiAqIGxvZygxICsgZXhwKC14KSkgKyAoMSAtIHopICogKHggKyBsb2coMSArIGV4cCgteCkpXG4gICAqICAgPSAoMSAtIHopICogeCArIGxvZygxICsgZXhwKC14KSlcbiAgICogICA9IHggLSB4ICogeiArIGxvZygxICsgZXhwKC14KSlcbiAgICpcbiAgICogICBGb3IgeCA8IDAsIHRvIGF2b2lkIG92ZXJmbG93IGluIGV4cCgteCksIHdlIHJlZm9ybXVsYXRlIHRoZSBhYm92ZVxuICAgKiAgICAgeCAtIHggKiB6ICsgbG9nKDEgKyBleHAoLXgpKVxuICAgKiAgID0gbG9nKGV4cCh4KSkgLSB4ICogeiArIGxvZygxICsgZXhwKC14KSlcbiAgICogICA9IC0geCAqIHogKyBsb2coMSArIGV4cCh4KSlcbiAgICpcbiAgICogSGVuY2UsIHRvIGVuc3VyZSBzdGFiaWxpdHkgYW5kIGF2b2lkIG92ZXJmbG93LCB0aGUgaW1wbGVtZW50YXRpb24gdXNlc1xuICAgKiB0aGlzIGVxdWl2YWxlbnQgZm9ybXVsYXRpb246XG4gICAqICAgICBtYXgoeCwgMCkgLSB4ICogeiArIGxvZygxICsgZXhwKC1hYnMoeCkpKVxuICAgKi9cbiAgY29uc3QgbWF4T3V0cHV0ID0gcmVsdSgkbG9naXRzKTtcbiAgY29uc3Qgb3V0cHV0WFRhcmdldCA9IG11bCgkbG9naXRzLCAkbGFiZWxzKTtcbiAgY29uc3Qgc2lnbW9pZE91dHB1dCA9IGxvZzFwKGV4cChuZWcoYWJzKCRsb2dpdHMpKSkpO1xuXG4gIHJldHVybiBhZGQoc3ViKG1heE91dHB1dCwgb3V0cHV0WFRhcmdldCksIHNpZ21vaWRPdXRwdXQpO1xufVxuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBzaWdtb2lkIGNyb3NzIGVudHJvcHkgbG9zcyBiZXR3ZWVuIHR3byB0ZW5zb3JzLlxuICpcbiAqIElmIGxhYmVsU21vb3RoaW5nIGlzIG5vbnplcm8sIHNtb290aCB0aGUgbGFiZWxzIHRvd2FyZHMgMS8yOlxuICpcbiAqICAgbmV3TXVsdGljbGFzc0xhYmVscyA9IG11bHRpY2xhc3NMYWJlbHMgKiAoMSAtIGxhYmVsU21vb3RoaW5nKVxuICogICAgICAgICAgICAgICAgICAgICAgICAgKyAwLjUgKiBsYWJlbFNtb290aGluZ1xuICpcbiAqIEBwYXJhbSBtdWx0aUNsYXNzTGFiZWxzIFRoZSBncm91bmQgdHJ1dGggb3V0cHV0IHRlbnNvciBvZiBzaGFwZVxuICogW2JhdGNoX3NpemUsIG51bV9jbGFzc2VzXSwgc2FtZSBkaW1lbnNpb25zIGFzICdwcmVkaWN0aW9ucycuXG4gKiBAcGFyYW0gbG9naXRzIFRoZSBwcmVkaWN0ZWQgb3V0cHV0cy5cbiAqIEBwYXJhbSB3ZWlnaHRzIFRlbnNvciB3aG9zZSByYW5rIGlzIGVpdGhlciAwLCBvciB0aGUgc2FtZSByYW5rIGFzXG4gKiAgICBgbGFiZWxzYCwgYW5kIG11c3QgYmUgYnJvYWRjYXN0YWJsZSB0byBgbGFiZWxzYCAoaS5lLiwgYWxsIGRpbWVuc2lvbnNcbiAqICAgIG11c3QgYmUgZWl0aGVyIGAxYCwgb3IgdGhlIHNhbWUgYXMgdGhlIGNvcnJlc3BvbmRpbmcgYGxvc3Nlc2BcbiAqICAgIGRpbWVuc2lvbikuXG4gKiBAcGFyYW0gbGFiZWxTbW9vdGhpbmcgSWYgZ3JlYXRlciB0aGFuIDAsIHRoZW4gc21vb3RoIHRoZSBsYWJlbHMuXG4gKiBAcGFyYW0gcmVkdWN0aW9uIFR5cGUgb2YgcmVkdWN0aW9uIHRvIGFwcGx5IHRvIGxvc3MuIFNob3VsZCBiZSBvZiB0eXBlXG4gKiAgICBgUmVkdWN0aW9uYFxuICpcbiAqIEBkb2MgeyBoZWFkaW5nOiAnVHJhaW5pbmcnLCBzdWJoZWFkaW5nOiAnTG9zc2VzJywgbmFtZXNwYWNlOiAnbG9zc2VzJyB9XG4gKi9cbmZ1bmN0aW9uIHNpZ21vaWRDcm9zc0VudHJvcHlfPFQgZXh0ZW5kcyBUZW5zb3IsIE8gZXh0ZW5kcyBUZW5zb3I+KFxuICAgIG11bHRpQ2xhc3NMYWJlbHM6IFR8VGVuc29yTGlrZSwgbG9naXRzOiBUfFRlbnNvckxpa2UsXG4gICAgd2VpZ2h0cz86IFRlbnNvcnxUZW5zb3JMaWtlLCBsYWJlbFNtb290aGluZyA9IDAsXG4gICAgcmVkdWN0aW9uID0gUmVkdWN0aW9uLlNVTV9CWV9OT05aRVJPX1dFSUdIVFMpOiBPIHtcbiAgbGV0ICRtdWx0aUNsYXNzTGFiZWxzID0gY29udmVydFRvVGVuc29yKFxuICAgICAgbXVsdGlDbGFzc0xhYmVscywgJ211bHRpQ2xhc3NMYWJlbHMnLCAnc2lnbW9pZENyb3NzRW50cm9weScpO1xuICBjb25zdCAkbG9naXRzID0gY29udmVydFRvVGVuc29yKGxvZ2l0cywgJ2xvZ2l0cycsICdzaWdtb2lkQ3Jvc3NFbnRyb3B5Jyk7XG4gIGxldCAkd2VpZ2h0czogVGVuc29yID0gbnVsbDtcbiAgaWYgKHdlaWdodHMgIT0gbnVsbCkge1xuICAgICR3ZWlnaHRzID0gY29udmVydFRvVGVuc29yKHdlaWdodHMsICd3ZWlnaHRzJywgJ3NpZ21vaWRDcm9zc0VudHJvcHknKTtcbiAgfVxuICBhc3NlcnRTaGFwZXNNYXRjaChcbiAgICAgICRtdWx0aUNsYXNzTGFiZWxzLnNoYXBlLCAkbG9naXRzLnNoYXBlLCAnRXJyb3IgaW4gc2lnbW9pZENyb3NzRW50cm9weTogJyk7XG5cbiAgaWYgKGxhYmVsU21vb3RoaW5nID4gMCkge1xuICAgIGNvbnN0IGxhYmVsU21vb3RoaW5nU2NhbGFyID0gc2NhbGFyKGxhYmVsU21vb3RoaW5nKTtcbiAgICBjb25zdCBvbmUgPSBzY2FsYXIoMSk7XG4gICAgY29uc3QgaGFsZiA9IHNjYWxhcigwLjUpO1xuXG4gICAgJG11bHRpQ2xhc3NMYWJlbHMgPVxuICAgICAgICBhZGQobXVsKCRtdWx0aUNsYXNzTGFiZWxzLCBzdWIob25lLCBsYWJlbFNtb290aGluZ1NjYWxhcikpLFxuICAgICAgICAgICAgbXVsKGhhbGYsIGxhYmVsU21vb3RoaW5nU2NhbGFyKSk7XG4gIH1cbiAgY29uc3QgbG9zc2VzID0gc2lnbW9pZENyb3NzRW50cm9weVdpdGhMb2dpdHNfKCRtdWx0aUNsYXNzTGFiZWxzLCAkbG9naXRzKTtcblxuICByZXR1cm4gY29tcHV0ZVdlaWdodGVkTG9zcyhsb3NzZXMsICR3ZWlnaHRzLCByZWR1Y3Rpb24pO1xufVxuXG5leHBvcnQgY29uc3Qgc2lnbW9pZENyb3NzRW50cm9weSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7c2lnbW9pZENyb3NzRW50cm9weV99KTtcbiJdfQ==