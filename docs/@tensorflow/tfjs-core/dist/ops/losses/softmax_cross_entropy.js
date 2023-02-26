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
import { customGrad } from '../../gradients';
import { convertToTensor } from '../../tensor_util_env';
import { assertShapesMatch } from '../../util';
import { add } from '../add';
import { expandShapeToKeepDim } from '../axis_util';
import { cast } from '../cast';
import { div } from '../div';
import { exp } from '../exp';
import { logSumExp } from '../log_sum_exp';
import { Reduction } from '../loss_ops_utils';
import { mul } from '../mul';
import { neg } from '../neg';
import { op } from '../operation';
import { reshape } from '../reshape';
import { scalar } from '../scalar';
import { sub } from '../sub';
import { sum } from '../sum';
import { computeWeightedLoss } from './compute_weighted_loss';
/**
 * Computes softmax cross entropy between logits and labels.
 *
 * Measures the probability error in discrete classification tasks in which
 * the classes are mutually exclusive (each entry is in exactly one class).
 * For example, each CIFAR-10 image is labeled with one and only one label: an
 * image can be a dog or a truck, but not both.
 *
 * `NOTE`: While the classes are mutually exclusive, their probabilities need
 * not be. All that is required is that each row of labels is a valid
 * probability distribution. If they are not, the computation of the gradient
 * will be incorrect.
 *
 * `WARNING`: This op expects unscaled logits, since it performs a softmax on
 * logits internally for efficiency. Do not call this op with the output of
 * softmax, as it will produce incorrect results.
 *
 * logits and labels must have the same shape, e.g. [batch_size, num_classes]
 * and the same dtype.
 * @param labels The labels array.
 * @param logits The logits array.
 * @param dim The dimension softmax would be performed on. Defaults to `-1`
 *     which indicates the last dimension.
 */
function softmaxCrossEntropyWithLogits_(labels, logits, dim = -1) {
    if (dim === -1) {
        dim = logits.rank - 1;
    }
    if (dim !== logits.rank - 1) {
        throw Error(`Softmax cross entropy along a non-last dimension is not yet ` +
            `supported. Labels / logits was rank ${logits.rank} ` +
            `and dim was ${dim}`);
    }
    // Use a custom gradient for numerical stability.
    const customOp = customGrad((labels, logits, save) => {
        // Reference:
        //   1. http://cs231n.github.io/linear-classify/#softmax
        //   2. https://blog.feedly.com/tricks-of-the-trade-logsumexp/
        const keepDims = true;
        const lse = logSumExp(logits, [dim], keepDims);
        const logResult = sub(cast(logits, 'float32'), lse);
        save([labels, logResult]);
        const costVector = neg(mul(logResult, labels));
        const value = sum(costVector, [dim]);
        const gradFunc = (dy, saved) => {
            const [labels, logResult] = saved;
            const dyShape = expandShapeToKeepDim(dy.shape, [dim]);
            return [
                mul(reshape(dy, dyShape), sub(cast(labels, 'float32'), exp(logResult))),
                mul(reshape(dy, dyShape), sub(exp(logResult), cast(labels, 'float32'))),
            ];
        };
        return { value, gradFunc };
    });
    return customOp(labels, logits);
}
/**
 * Computes the softmax cross entropy loss between two tensors.
 *
 * If labelSmoothing is nonzero, smooth the labels towards 1/2:
 *
 *   newOnehotLabels = onehotLabels * (1 - labelSmoothing)
 *                         + labelSmoothing / numClasses
 *
 * @param onehotLabels One hot encoded labels
 *    [batch_size, num_classes], same dimensions as 'predictions'.
 * @param logits The predicted outputs.
 * @param weights Tensor whose rank is either 0, or 1, and must be
 *    broadcastable to `loss`  of shape [batch_size]
 * @param labelSmoothing If greater than 0, then smooth the labels.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' }
 */
function softmaxCrossEntropy_(onehotLabels, logits, weights, labelSmoothing = 0, reduction = Reduction.SUM_BY_NONZERO_WEIGHTS) {
    let $onehotLabels = convertToTensor(onehotLabels, 'onehotLabels', 'softmaxCrossEntropy');
    const $logits = convertToTensor(logits, 'logits', 'softmaxCrossEntropy');
    let $weights = null;
    if (weights != null) {
        $weights = convertToTensor(weights, 'weights', 'softmaxCrossEntropy');
    }
    assertShapesMatch($onehotLabels.shape, $logits.shape, 'Error in softmaxCrossEntropy: ');
    if (labelSmoothing > 0) {
        const labelSmoothingScalar = scalar(labelSmoothing);
        const one = scalar(1);
        const numClasses = scalar($onehotLabels.shape[1]);
        $onehotLabels =
            add(mul($onehotLabels, sub(one, labelSmoothingScalar)), div(labelSmoothingScalar, numClasses));
    }
    const losses = softmaxCrossEntropyWithLogits_($onehotLabels, $logits);
    return computeWeightedLoss(losses, $weights, reduction);
}
export const softmaxCrossEntropy = /* @__PURE__ */ op({ softmaxCrossEntropy_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic29mdG1heF9jcm9zc19lbnRyb3B5LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvbG9zc2VzL3NvZnRtYXhfY3Jvc3NfZW50cm9weS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFHM0MsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXRELE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUM3QyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxvQkFBb0IsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNsRCxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBQzdCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDM0IsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUMzQixPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDekMsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQzVDLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDM0IsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUMzQixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ2hDLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDbkMsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENB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QixRQUFRLEdBQUcsZUFBZSxDQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUscUJBQXFCLENBQUMsQ0FBQztLQUN2RTtJQUVELGlCQUFpQixDQUNiLGFBQWEsQ0FBQyxLQUFLLEVBQUUsT0FBTyxDQUFDLEtBQUssRUFBRSxnQ0FBZ0MsQ0FBQyxDQUFDO0lBRTFFLElBQUksY0FBYyxHQUFHLENBQUMsRUFBRTtRQUN0QixNQUFNLG9CQUFvQixHQUFHLE1BQU0sQ0FBQyxjQUFjLENBQUMsQ0FBQztRQUNwRCxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdEIsTUFBTSxVQUFVLEdBQUcsTUFBTSxDQUFDLGFBQWEsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUVsRCxhQUFhO1lBQ1QsR0FBRyxDQUFDLEdBQUcsQ0FBQyxhQUFhLEVBQUUsR0FBRyxDQUFDLEdBQUcsRUFBRSxvQkFBb0IsQ0FBQyxDQUFDLEVBQ2xELEdBQUcsQ0FBQyxvQkFBb0IsRUFBRSxVQUFVLENBQUMsQ0FBQyxDQUFDO0tBQ2hEO0lBRUQsTUFBTSxNQUFNLEdBQUcsOEJBQThCLENBQUMsYUFBYSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBRXRFLE9BQU8sbUJBQW1CLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxTQUFTLENBQUMsQ0FBQztBQUMxRCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sbUJBQW1CLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLG9CQUFvQixFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7Y3VzdG9tR3JhZH0gZnJvbSAnLi4vLi4vZ3JhZGllbnRzJztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi8uLi90ZW5zb3InO1xuaW1wb3J0IHtHcmFkU2F2ZUZ1bmN9IGZyb20gJy4uLy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0IHthc3NlcnRTaGFwZXNNYXRjaH0gZnJvbSAnLi4vLi4vdXRpbCc7XG5pbXBvcnQge2FkZH0gZnJvbSAnLi4vYWRkJztcbmltcG9ydCB7ZXhwYW5kU2hhcGVUb0tlZXBEaW19IGZyb20gJy4uL2F4aXNfdXRpbCc7XG5pbXBvcnQge2Nhc3R9IGZyb20gJy4uL2Nhc3QnO1xuaW1wb3J0IHtkaXZ9IGZyb20gJy4uL2Rpdic7XG5pbXBvcnQge2V4cH0gZnJvbSAnLi4vZXhwJztcbmltcG9ydCB7bG9nU3VtRXhwfSBmcm9tICcuLi9sb2dfc3VtX2V4cCc7XG5pbXBvcnQge1JlZHVjdGlvbn0gZnJvbSAnLi4vbG9zc19vcHNfdXRpbHMnO1xuaW1wb3J0IHttdWx9IGZyb20gJy4uL211bCc7XG5pbXBvcnQge25lZ30gZnJvbSAnLi4vbmVnJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4uL3Jlc2hhcGUnO1xuaW1wb3J0IHtzY2FsYXJ9IGZyb20gJy4uL3NjYWxhcic7XG5pbXBvcnQge3N1Yn0gZnJvbSAnLi4vc3ViJztcbmltcG9ydCB7c3VtfSBmcm9tICcuLi9zdW0nO1xuXG5pbXBvcnQge2NvbXB1dGVXZWlnaHRlZExvc3N9IGZyb20gJy4vY29tcHV0ZV93ZWlnaHRlZF9sb3NzJztcblxuLyoqXG4gKiBDb21wdXRlcyBzb2Z0bWF4IGNyb3NzIGVudHJvcHkgYmV0d2VlbiBsb2dpdHMgYW5kIGxhYmVscy5cbiAqXG4gKiBNZWFzdXJlcyB0aGUgcHJvYmFiaWxpdHkgZXJyb3IgaW4gZGlzY3JldGUgY2xhc3NpZmljYXRpb24gdGFza3MgaW4gd2hpY2hcbiAqIHRoZSBjbGFzc2VzIGFyZSBtdXR1YWxseSBleGNsdXNpdmUgKGVhY2ggZW50cnkgaXMgaW4gZXhhY3RseSBvbmUgY2xhc3MpLlxuICogRm9yIGV4YW1wbGUsIGVhY2ggQ0lGQVItMTAgaW1hZ2UgaXMgbGFiZWxlZCB3aXRoIG9uZSBhbmQgb25seSBvbmUgbGFiZWw6IGFuXG4gKiBpbWFnZSBjYW4gYmUgYSBkb2cgb3IgYSB0cnVjaywgYnV0IG5vdCBib3RoLlxuICpcbiAqIGBOT1RFYDogV2hpbGUgdGhlIGNsYXNzZXMgYXJlIG11dHVhbGx5IGV4Y2x1c2l2ZSwgdGhlaXIgcHJvYmFiaWxpdGllcyBuZWVkXG4gKiBub3QgYmUuIEFsbCB0aGF0IGlzIHJlcXVpcmVkIGlzIHRoYXQgZWFjaCByb3cgb2YgbGFiZWxzIGlzIGEgdmFsaWRcbiAqIHByb2JhYmlsaXR5IGRpc3RyaWJ1dGlvbi4gSWYgdGhleSBhcmUgbm90LCB0aGUgY29tcHV0YXRpb24gb2YgdGhlIGdyYWRpZW50XG4gKiB3aWxsIGJlIGluY29ycmVjdC5cbiAqXG4gKiBgV0FSTklOR2A6IFRoaXMgb3AgZXhwZWN0cyB1bnNjYWxlZCBsb2dpdHMsIHNpbmNlIGl0IHBlcmZvcm1zIGEgc29mdG1heCBvblxuICogbG9naXRzIGludGVybmFsbHkgZm9yIGVmZmljaWVuY3kuIERvIG5vdCBjYWxsIHRoaXMgb3Agd2l0aCB0aGUgb3V0cHV0IG9mXG4gKiBzb2Z0bWF4LCBhcyBpdCB3aWxsIHByb2R1Y2UgaW5jb3JyZWN0IHJlc3VsdHMuXG4gKlxuICogbG9naXRzIGFuZCBsYWJlbHMgbXVzdCBoYXZlIHRoZSBzYW1lIHNoYXBlLCBlLmcuIFtiYXRjaF9zaXplLCBudW1fY2xhc3Nlc11cbiAqIGFuZCB0aGUgc2FtZSBkdHlwZS5cbiAqIEBwYXJhbSBsYWJlbHMgVGhlIGxhYmVscyBhcnJheS5cbiAqIEBwYXJhbSBsb2dpdHMgVGhlIGxvZ2l0cyBhcnJheS5cbiAqIEBwYXJhbSBkaW0gVGhlIGRpbWVuc2lvbiBzb2Z0bWF4IHdvdWxkIGJlIHBlcmZvcm1lZCBvbi4gRGVmYXVsdHMgdG8gYC0xYFxuICogICAgIHdoaWNoIGluZGljYXRlcyB0aGUgbGFzdCBkaW1lbnNpb24uXG4gKi9cbmZ1bmN0aW9uIHNvZnRtYXhDcm9zc0VudHJvcHlXaXRoTG9naXRzXzxUIGV4dGVuZHMgVGVuc29yLCBPIGV4dGVuZHMgVGVuc29yPihcbiAgICBsYWJlbHM6IFQsIGxvZ2l0czogVCwgZGltID0gLTEpOiBPIHtcbiAgaWYgKGRpbSA9PT0gLTEpIHtcbiAgICBkaW0gPSBsb2dpdHMucmFuayAtIDE7XG4gIH1cblxuICBpZiAoZGltICE9PSBsb2dpdHMucmFuayAtIDEpIHtcbiAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgYFNvZnRtYXggY3Jvc3MgZW50cm9weSBhbG9uZyBhIG5vbi1sYXN0IGRpbWVuc2lvbiBpcyBub3QgeWV0IGAgK1xuICAgICAgICBgc3VwcG9ydGVkLiBMYWJlbHMgLyBsb2dpdHMgd2FzIHJhbmsgJHtsb2dpdHMucmFua30gYCArXG4gICAgICAgIGBhbmQgZGltIHdhcyAke2RpbX1gKTtcbiAgfVxuICAvLyBVc2UgYSBjdXN0b20gZ3JhZGllbnQgZm9yIG51bWVyaWNhbCBzdGFiaWxpdHkuXG4gIGNvbnN0IGN1c3RvbU9wID1cbiAgICAgIGN1c3RvbUdyYWQoKGxhYmVsczogVGVuc29yLCBsb2dpdHM6IFRlbnNvciwgc2F2ZTogR3JhZFNhdmVGdW5jKSA9PiB7XG4gICAgICAgIC8vIFJlZmVyZW5jZTpcbiAgICAgICAgLy8gICAxLiBodHRwOi8vY3MyMzFuLmdpdGh1Yi5pby9saW5lYXItY2xhc3NpZnkvI3NvZnRtYXhcbiAgICAgICAgLy8gICAyLiBodHRwczovL2Jsb2cuZmVlZGx5LmNvbS90cmlja3Mtb2YtdGhlLXRyYWRlLWxvZ3N1bWV4cC9cbiAgICAgICAgY29uc3Qga2VlcERpbXMgPSB0cnVlO1xuICAgICAgICBjb25zdCBsc2UgPSBsb2dTdW1FeHAobG9naXRzLCBbZGltXSwga2VlcERpbXMpO1xuICAgICAgICBjb25zdCBsb2dSZXN1bHQgPSBzdWIoY2FzdChsb2dpdHMsICdmbG9hdDMyJyksIGxzZSk7XG4gICAgICAgIHNhdmUoW2xhYmVscywgbG9nUmVzdWx0XSk7XG5cbiAgICAgICAgY29uc3QgY29zdFZlY3RvciA9IG5lZyhtdWwobG9nUmVzdWx0LCBsYWJlbHMpKTtcbiAgICAgICAgY29uc3QgdmFsdWU6IE8gPSBzdW0oY29zdFZlY3RvciwgW2RpbV0pO1xuXG4gICAgICAgIGNvbnN0IGdyYWRGdW5jID0gKGR5OiBPLCBzYXZlZDogVGVuc29yW10pID0+IHtcbiAgICAgICAgICBjb25zdCBbbGFiZWxzLCBsb2dSZXN1bHRdID0gc2F2ZWQ7XG4gICAgICAgICAgY29uc3QgZHlTaGFwZSA9IGV4cGFuZFNoYXBlVG9LZWVwRGltKGR5LnNoYXBlLCBbZGltXSk7XG4gICAgICAgICAgcmV0dXJuIFtcbiAgICAgICAgICAgIG11bChyZXNoYXBlKGR5LCBkeVNoYXBlKSxcbiAgICAgICAgICAgICAgICBzdWIoY2FzdChsYWJlbHMsICdmbG9hdDMyJyksIGV4cChsb2dSZXN1bHQpKSksXG4gICAgICAgICAgICBtdWwocmVzaGFwZShkeSwgZHlTaGFwZSksXG4gICAgICAgICAgICAgICAgc3ViKGV4cChsb2dSZXN1bHQpLCBjYXN0KGxhYmVscywgJ2Zsb2F0MzInKSkpLFxuICAgICAgICAgIF07XG4gICAgICAgIH07XG4gICAgICAgIHJldHVybiB7dmFsdWUsIGdyYWRGdW5jfTtcbiAgICAgIH0pO1xuXG4gIHJldHVybiBjdXN0b21PcChsYWJlbHMsIGxvZ2l0cyk7XG59XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIHNvZnRtYXggY3Jvc3MgZW50cm9weSBsb3NzIGJldHdlZW4gdHdvIHRlbnNvcnMuXG4gKlxuICogSWYgbGFiZWxTbW9vdGhpbmcgaXMgbm9uemVybywgc21vb3RoIHRoZSBsYWJlbHMgdG93YXJkcyAxLzI6XG4gKlxuICogICBuZXdPbmVob3RMYWJlbHMgPSBvbmVob3RMYWJlbHMgKiAoMSAtIGxhYmVsU21vb3RoaW5nKVxuICogICAgICAgICAgICAgICAgICAgICAgICAgKyBsYWJlbFNtb290aGluZyAvIG51bUNsYXNzZXNcbiAqXG4gKiBAcGFyYW0gb25laG90TGFiZWxzIE9uZSBob3QgZW5jb2RlZCBsYWJlbHNcbiAqICAgIFtiYXRjaF9zaXplLCBudW1fY2xhc3Nlc10sIHNhbWUgZGltZW5zaW9ucyBhcyAncHJlZGljdGlvbnMnLlxuICogQHBhcmFtIGxvZ2l0cyBUaGUgcHJlZGljdGVkIG91dHB1dHMuXG4gKiBAcGFyYW0gd2VpZ2h0cyBUZW5zb3Igd2hvc2UgcmFuayBpcyBlaXRoZXIgMCwgb3IgMSwgYW5kIG11c3QgYmVcbiAqICAgIGJyb2FkY2FzdGFibGUgdG8gYGxvc3NgICBvZiBzaGFwZSBbYmF0Y2hfc2l6ZV1cbiAqIEBwYXJhbSBsYWJlbFNtb290aGluZyBJZiBncmVhdGVyIHRoYW4gMCwgdGhlbiBzbW9vdGggdGhlIGxhYmVscy5cbiAqIEBwYXJhbSByZWR1Y3Rpb24gVHlwZSBvZiByZWR1Y3Rpb24gdG8gYXBwbHkgdG8gbG9zcy4gU2hvdWxkIGJlIG9mIHR5cGVcbiAqICAgIGBSZWR1Y3Rpb25gXG4gKlxuICogQGRvYyB7IGhlYWRpbmc6ICdUcmFpbmluZycsIHN1YmhlYWRpbmc6ICdMb3NzZXMnLCBuYW1lc3BhY2U6ICdsb3NzZXMnIH1cbiAqL1xuZnVuY3Rpb24gc29mdG1heENyb3NzRW50cm9weV88VCBleHRlbmRzIFRlbnNvciwgTyBleHRlbmRzIFRlbnNvcj4oXG4gICAgb25laG90TGFiZWxzOiBUfFRlbnNvckxpa2UsIGxvZ2l0czogVHxUZW5zb3JMaWtlLFxuICAgIHdlaWdodHM/OiBUZW5zb3J8VGVuc29yTGlrZSwgbGFiZWxTbW9vdGhpbmcgPSAwLFxuICAgIHJlZHVjdGlvbiA9IFJlZHVjdGlvbi5TVU1fQllfTk9OWkVST19XRUlHSFRTKTogTyB7XG4gIGxldCAkb25laG90TGFiZWxzID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihvbmVob3RMYWJlbHMsICdvbmVob3RMYWJlbHMnLCAnc29mdG1heENyb3NzRW50cm9weScpO1xuICBjb25zdCAkbG9naXRzID0gY29udmVydFRvVGVuc29yKGxvZ2l0cywgJ2xvZ2l0cycsICdzb2Z0bWF4Q3Jvc3NFbnRyb3B5Jyk7XG4gIGxldCAkd2VpZ2h0czogVGVuc29yID0gbnVsbDtcblxuICBpZiAod2VpZ2h0cyAhPSBudWxsKSB7XG4gICAgJHdlaWdodHMgPSBjb252ZXJ0VG9UZW5zb3Iod2VpZ2h0cywgJ3dlaWdodHMnLCAnc29mdG1heENyb3NzRW50cm9weScpO1xuICB9XG5cbiAgYXNzZXJ0U2hhcGVzTWF0Y2goXG4gICAgICAkb25laG90TGFiZWxzLnNoYXBlLCAkbG9naXRzLnNoYXBlLCAnRXJyb3IgaW4gc29mdG1heENyb3NzRW50cm9weTogJyk7XG5cbiAgaWYgKGxhYmVsU21vb3RoaW5nID4gMCkge1xuICAgIGNvbnN0IGxhYmVsU21vb3RoaW5nU2NhbGFyID0gc2NhbGFyKGxhYmVsU21vb3RoaW5nKTtcbiAgICBjb25zdCBvbmUgPSBzY2FsYXIoMSk7XG4gICAgY29uc3QgbnVtQ2xhc3NlcyA9IHNjYWxhcigkb25laG90TGFiZWxzLnNoYXBlWzFdKTtcblxuICAgICRvbmVob3RMYWJlbHMgPVxuICAgICAgICBhZGQobXVsKCRvbmVob3RMYWJlbHMsIHN1YihvbmUsIGxhYmVsU21vb3RoaW5nU2NhbGFyKSksXG4gICAgICAgICAgICBkaXYobGFiZWxTbW9vdGhpbmdTY2FsYXIsIG51bUNsYXNzZXMpKTtcbiAgfVxuXG4gIGNvbnN0IGxvc3NlcyA9IHNvZnRtYXhDcm9zc0VudHJvcHlXaXRoTG9naXRzXygkb25laG90TGFiZWxzLCAkbG9naXRzKTtcblxuICByZXR1cm4gY29tcHV0ZVdlaWdodGVkTG9zcyhsb3NzZXMsICR3ZWlnaHRzLCByZWR1Y3Rpb24pO1xufVxuXG5leHBvcnQgY29uc3Qgc29mdG1heENyb3NzRW50cm9weSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7c29mdG1heENyb3NzRW50cm9weV99KTtcbiJdfQ==