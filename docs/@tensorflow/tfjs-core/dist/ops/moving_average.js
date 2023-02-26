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
import { assertTypesMatch } from '../tensor_util';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { add } from './add';
import { div } from './div';
import { mul } from './mul';
import { op } from './operation';
import { pow } from './pow';
import { scalar } from './scalar';
import { sub } from './sub';
/**
 * Compute the moving average of a variable.
 *
 * Without zeroDebias, the moving average operation is defined by:
 *   `v += delta`
 * where
 *   `delta = (1 - decay) * (x - v)`
 *
 * With zeroDebias (default), the `delta` term is scaled to debias the
 * effect of the (assumed) zero-initialization of `v`.
 *   `delta /= (1 - decay ^ step)`
 *
 * For more details on the zero-debiasing algorithm, see:
 *   https://arxiv.org/abs/1412.6980
 *
 * Note that this function is completely stateless and does not keep track of
 * step count. The step count needs to be maintained by the caller and passed
 * in as `step`.
 *
 * @param v The current moving average value.
 * @param x New input value, must have the same shape and dtype as `v`.
 * @param decay The decay factor. Typical values are 0.95 and 0.99.
 * @param step Step count.
 * @param zeroDebias: Whether zeroDebias is to be performed (default: `true`).
 * @returns The new moving average value.
 *
 * @doc {heading: 'Operations', subheading: 'Moving Average'}
 */
function movingAverage_(v, x, decay, step, zeroDebias = true) {
    const $v = convertToTensor(v, 'v', 'movingAverage');
    const $x = convertToTensor(x, 'x', 'movingAverage');
    const $decay = convertToTensor(decay, 'decay', 'movingAverage');
    assertTypesMatch($v, $x);
    util.assert(util.arraysEqual($v.shape, $x.shape), () => 'Shape mismatch in v and x');
    const one = scalar(1);
    const oneMinusDecay = sub(one, $decay);
    let update = mul(sub($x, $v), oneMinusDecay);
    if (zeroDebias) {
        util.assert(step != null, () => 'When using zeroDebias: true, step is required.');
        const $step = convertToTensor(step, 'step', 'movingAverage');
        update = div(update, sub(one, pow($decay, $step)));
    }
    return add($v, update);
}
export const movingAverage = /* @__PURE__ */ op({ movingAverage_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibW92aW5nX2F2ZXJhZ2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9tb3ZpbmdfYXZlcmFnZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUNoRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDaEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUUxQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBMkJHO0FBQ0gsU0FBUyxjQUFjLENBQ25CLENBQWUsRUFBRSxDQUFlLEVBQUUsS0FBb0IsRUFDdEQsSUFBb0IsRUFBRSxVQUFVLEdBQUcsSUFBSTtJQUN6QyxNQUFNLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxlQUFlLENBQUMsQ0FBQztJQUNwRCxNQUFNLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxlQUFlLENBQUMsQ0FBQztJQUNwRCxNQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxlQUFlLENBQUMsQ0FBQztJQUVoRSxnQkFBZ0IsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7SUFDekIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsV0FBVyxDQUFDLEVBQUUsQ0FBQyxLQUFLLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLDJCQUEyQixDQUFDLENBQUM7SUFFN0UsTUFBTSxHQUFHLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3RCLE1BQU0sYUFBYSxHQUFHLEdBQUcsQ0FBQyxHQUFHLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFFdkMsSUFBSSxNQUFNLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsYUFBYSxDQUFDLENBQUM7SUFDN0MsSUFBSSxVQUFVLEVBQUU7UUFDZCxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksSUFBSSxJQUFJLEVBQUUsR0FBRyxFQUFFLENBQUMsZ0RBQWdELENBQUMsQ0FBQztRQUMxRSxNQUFNLEtBQUssR0FBRyxlQUFlLENBQUMsSUFBSSxFQUFFLE1BQU0sRUFBRSxlQUFlLENBQUMsQ0FBQztRQUM3RCxNQUFNLEdBQUcsR0FBRyxDQUFDLE1BQU0sRUFBRSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxNQUFNLEVBQUUsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQ3BEO0lBQ0QsT0FBTyxHQUFHLENBQUMsRUFBRSxFQUFFLE1BQU0sQ0FBQyxDQUFDO0FBQ3pCLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxhQUFhLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLGNBQWMsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7U2NhbGFyLCBUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge2Fzc2VydFR5cGVzTWF0Y2h9IGZyb20gJy4uL3RlbnNvcl91dGlsJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge2FkZH0gZnJvbSAnLi9hZGQnO1xuaW1wb3J0IHtkaXZ9IGZyb20gJy4vZGl2JztcbmltcG9ydCB7bXVsfSBmcm9tICcuL211bCc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Bvd30gZnJvbSAnLi9wb3cnO1xuaW1wb3J0IHtzY2FsYXJ9IGZyb20gJy4vc2NhbGFyJztcbmltcG9ydCB7c3VifSBmcm9tICcuL3N1Yic7XG5cbi8qKlxuICogQ29tcHV0ZSB0aGUgbW92aW5nIGF2ZXJhZ2Ugb2YgYSB2YXJpYWJsZS5cbiAqXG4gKiBXaXRob3V0IHplcm9EZWJpYXMsIHRoZSBtb3ZpbmcgYXZlcmFnZSBvcGVyYXRpb24gaXMgZGVmaW5lZCBieTpcbiAqICAgYHYgKz0gZGVsdGFgXG4gKiB3aGVyZVxuICogICBgZGVsdGEgPSAoMSAtIGRlY2F5KSAqICh4IC0gdilgXG4gKlxuICogV2l0aCB6ZXJvRGViaWFzIChkZWZhdWx0KSwgdGhlIGBkZWx0YWAgdGVybSBpcyBzY2FsZWQgdG8gZGViaWFzIHRoZVxuICogZWZmZWN0IG9mIHRoZSAoYXNzdW1lZCkgemVyby1pbml0aWFsaXphdGlvbiBvZiBgdmAuXG4gKiAgIGBkZWx0YSAvPSAoMSAtIGRlY2F5IF4gc3RlcClgXG4gKlxuICogRm9yIG1vcmUgZGV0YWlscyBvbiB0aGUgemVyby1kZWJpYXNpbmcgYWxnb3JpdGhtLCBzZWU6XG4gKiAgIGh0dHBzOi8vYXJ4aXYub3JnL2Ficy8xNDEyLjY5ODBcbiAqXG4gKiBOb3RlIHRoYXQgdGhpcyBmdW5jdGlvbiBpcyBjb21wbGV0ZWx5IHN0YXRlbGVzcyBhbmQgZG9lcyBub3Qga2VlcCB0cmFjayBvZlxuICogc3RlcCBjb3VudC4gVGhlIHN0ZXAgY291bnQgbmVlZHMgdG8gYmUgbWFpbnRhaW5lZCBieSB0aGUgY2FsbGVyIGFuZCBwYXNzZWRcbiAqIGluIGFzIGBzdGVwYC5cbiAqXG4gKiBAcGFyYW0gdiBUaGUgY3VycmVudCBtb3ZpbmcgYXZlcmFnZSB2YWx1ZS5cbiAqIEBwYXJhbSB4IE5ldyBpbnB1dCB2YWx1ZSwgbXVzdCBoYXZlIHRoZSBzYW1lIHNoYXBlIGFuZCBkdHlwZSBhcyBgdmAuXG4gKiBAcGFyYW0gZGVjYXkgVGhlIGRlY2F5IGZhY3Rvci4gVHlwaWNhbCB2YWx1ZXMgYXJlIDAuOTUgYW5kIDAuOTkuXG4gKiBAcGFyYW0gc3RlcCBTdGVwIGNvdW50LlxuICogQHBhcmFtIHplcm9EZWJpYXM6IFdoZXRoZXIgemVyb0RlYmlhcyBpcyB0byBiZSBwZXJmb3JtZWQgKGRlZmF1bHQ6IGB0cnVlYCkuXG4gKiBAcmV0dXJucyBUaGUgbmV3IG1vdmluZyBhdmVyYWdlIHZhbHVlLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ01vdmluZyBBdmVyYWdlJ31cbiAqL1xuZnVuY3Rpb24gbW92aW5nQXZlcmFnZV88VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgdjogVHxUZW5zb3JMaWtlLCB4OiBUfFRlbnNvckxpa2UsIGRlY2F5OiBudW1iZXJ8U2NhbGFyLFxuICAgIHN0ZXA/OiBudW1iZXJ8U2NhbGFyLCB6ZXJvRGViaWFzID0gdHJ1ZSk6IFQge1xuICBjb25zdCAkdiA9IGNvbnZlcnRUb1RlbnNvcih2LCAndicsICdtb3ZpbmdBdmVyYWdlJyk7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ21vdmluZ0F2ZXJhZ2UnKTtcbiAgY29uc3QgJGRlY2F5ID0gY29udmVydFRvVGVuc29yKGRlY2F5LCAnZGVjYXknLCAnbW92aW5nQXZlcmFnZScpO1xuXG4gIGFzc2VydFR5cGVzTWF0Y2goJHYsICR4KTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB1dGlsLmFycmF5c0VxdWFsKCR2LnNoYXBlLCAkeC5zaGFwZSksICgpID0+ICdTaGFwZSBtaXNtYXRjaCBpbiB2IGFuZCB4Jyk7XG5cbiAgY29uc3Qgb25lID0gc2NhbGFyKDEpO1xuICBjb25zdCBvbmVNaW51c0RlY2F5ID0gc3ViKG9uZSwgJGRlY2F5KTtcblxuICBsZXQgdXBkYXRlID0gbXVsKHN1YigkeCwgJHYpLCBvbmVNaW51c0RlY2F5KTtcbiAgaWYgKHplcm9EZWJpYXMpIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgc3RlcCAhPSBudWxsLCAoKSA9PiAnV2hlbiB1c2luZyB6ZXJvRGViaWFzOiB0cnVlLCBzdGVwIGlzIHJlcXVpcmVkLicpO1xuICAgIGNvbnN0ICRzdGVwID0gY29udmVydFRvVGVuc29yKHN0ZXAsICdzdGVwJywgJ21vdmluZ0F2ZXJhZ2UnKTtcbiAgICB1cGRhdGUgPSBkaXYodXBkYXRlLCBzdWIob25lLCBwb3coJGRlY2F5LCAkc3RlcCkpKTtcbiAgfVxuICByZXR1cm4gYWRkKCR2LCB1cGRhdGUpO1xufVxuXG5leHBvcnQgY29uc3QgbW92aW5nQXZlcmFnZSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7bW92aW5nQXZlcmFnZV99KTtcbiJdfQ==