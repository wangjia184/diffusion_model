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
import { Tensor } from '../tensor';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { add } from './add';
import { div } from './div';
import { getNoiseShape } from './dropout_util';
import { floor } from './floor';
import { mul } from './mul';
import { op } from './operation';
import { randomUniform } from './random_uniform';
/**
 * Computes dropout.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 2, 1]);
 * const rate = 0.75;
 * const output = tf.dropout(x, rate);
 * output.print();
 * ```
 *
 * @param x A floating point Tensor or TensorLike.
 * @param rate A float in the range [0, 1). The probability that each element
 *   of x is discarded.
 * @param noiseShape An array of numbers of type int32, representing the
 * shape for randomly generated keep/drop flags. If the noiseShape has null
 * value, it will be automatically replaced with the x's relative dimension
 * size. Optional.
 * @param seed Used to create random seeds. Optional.
 * @returns A Tensor of the same shape of x.
 *
 * @doc {heading: 'Operations', subheading: 'Dropout'}
 */
function dropout_(x, rate, noiseShape, seed) {
    const $x = convertToTensor(x, 'x', 'dropout');
    util.assert($x.dtype === 'float32', () => `x has to be a floating point tensor since it's going to be ` +
        `scaled, but got a ${$x.dtype} tensor instead.`);
    util.assert(rate >= 0 && rate < 1, () => `rate must be a float in the range [0, 1), but got ${rate}.`);
    if (rate === 0) {
        return x instanceof Tensor ? $x.clone() : $x;
    }
    const $noiseShape = getNoiseShape($x, noiseShape);
    const keepProb = 1 - rate;
    const multiplier = div(floor(add(randomUniform($noiseShape, 0, 1, 'float32', seed), keepProb)), keepProb);
    return mul($x, multiplier);
}
export const dropout = /* @__PURE__ */ op({ dropout_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZHJvcG91dC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2Ryb3BvdXQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUM3QyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBQzlCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsYUFBYSxFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFFL0M7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXFCRztBQUNILFNBQVMsUUFBUSxDQUNiLENBQW9CLEVBQUUsSUFBWSxFQUFFLFVBQXFCLEVBQ3pELElBQW9CO0lBQ3RCLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBRTlDLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLEtBQUssS0FBSyxTQUFTLEVBQ3RCLEdBQUcsRUFBRSxDQUFDLDZEQUE2RDtRQUMvRCxxQkFBcUIsRUFBRSxDQUFDLEtBQUssa0JBQWtCLENBQUMsQ0FBQztJQUN6RCxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksSUFBSSxDQUFDLElBQUksSUFBSSxHQUFHLENBQUMsRUFDckIsR0FBRyxFQUFFLENBQUMscURBQXFELElBQUksR0FBRyxDQUFDLENBQUM7SUFFeEUsSUFBSSxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ2QsT0FBTyxDQUFDLFlBQVksTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztLQUM5QztJQUVELE1BQU0sV0FBVyxHQUFHLGFBQWEsQ0FBQyxFQUFFLEVBQUUsVUFBVSxDQUFDLENBQUM7SUFDbEQsTUFBTSxRQUFRLEdBQUcsQ0FBQyxHQUFHLElBQUksQ0FBQztJQUMxQixNQUFNLFVBQVUsR0FBRyxHQUFHLENBQ2xCLEtBQUssQ0FBQyxHQUFHLENBQUMsYUFBYSxDQUFDLFdBQVcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLFNBQVMsRUFBRSxJQUFJLENBQUMsRUFBRSxRQUFRLENBQUMsQ0FBQyxFQUN2RSxRQUFRLENBQUMsQ0FBQztJQUVkLE9BQU8sR0FBRyxDQUFDLEVBQUUsRUFBRSxVQUFVLENBQUMsQ0FBQztBQUM3QixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxRQUFRLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge2FkZH0gZnJvbSAnLi9hZGQnO1xuaW1wb3J0IHtkaXZ9IGZyb20gJy4vZGl2JztcbmltcG9ydCB7Z2V0Tm9pc2VTaGFwZX0gZnJvbSAnLi9kcm9wb3V0X3V0aWwnO1xuaW1wb3J0IHtmbG9vcn0gZnJvbSAnLi9mbG9vcic7XG5pbXBvcnQge211bH0gZnJvbSAnLi9tdWwnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyYW5kb21Vbmlmb3JtfSBmcm9tICcuL3JhbmRvbV91bmlmb3JtJztcblxuLyoqXG4gKiBDb21wdXRlcyBkcm9wb3V0LlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDIsIDFdKTtcbiAqIGNvbnN0IHJhdGUgPSAwLjc1O1xuICogY29uc3Qgb3V0cHV0ID0gdGYuZHJvcG91dCh4LCByYXRlKTtcbiAqIG91dHB1dC5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHggQSBmbG9hdGluZyBwb2ludCBUZW5zb3Igb3IgVGVuc29yTGlrZS5cbiAqIEBwYXJhbSByYXRlIEEgZmxvYXQgaW4gdGhlIHJhbmdlIFswLCAxKS4gVGhlIHByb2JhYmlsaXR5IHRoYXQgZWFjaCBlbGVtZW50XG4gKiAgIG9mIHggaXMgZGlzY2FyZGVkLlxuICogQHBhcmFtIG5vaXNlU2hhcGUgQW4gYXJyYXkgb2YgbnVtYmVycyBvZiB0eXBlIGludDMyLCByZXByZXNlbnRpbmcgdGhlXG4gKiBzaGFwZSBmb3IgcmFuZG9tbHkgZ2VuZXJhdGVkIGtlZXAvZHJvcCBmbGFncy4gSWYgdGhlIG5vaXNlU2hhcGUgaGFzIG51bGxcbiAqIHZhbHVlLCBpdCB3aWxsIGJlIGF1dG9tYXRpY2FsbHkgcmVwbGFjZWQgd2l0aCB0aGUgeCdzIHJlbGF0aXZlIGRpbWVuc2lvblxuICogc2l6ZS4gT3B0aW9uYWwuXG4gKiBAcGFyYW0gc2VlZCBVc2VkIHRvIGNyZWF0ZSByYW5kb20gc2VlZHMuIE9wdGlvbmFsLlxuICogQHJldHVybnMgQSBUZW5zb3Igb2YgdGhlIHNhbWUgc2hhcGUgb2YgeC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdEcm9wb3V0J31cbiAqL1xuZnVuY3Rpb24gZHJvcG91dF8oXG4gICAgeDogVGVuc29yfFRlbnNvckxpa2UsIHJhdGU6IG51bWJlciwgbm9pc2VTaGFwZT86IG51bWJlcltdLFxuICAgIHNlZWQ/OiBudW1iZXJ8c3RyaW5nKTogVGVuc29yIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnZHJvcG91dCcpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHguZHR5cGUgPT09ICdmbG9hdDMyJyxcbiAgICAgICgpID0+IGB4IGhhcyB0byBiZSBhIGZsb2F0aW5nIHBvaW50IHRlbnNvciBzaW5jZSBpdCdzIGdvaW5nIHRvIGJlIGAgK1xuICAgICAgICAgIGBzY2FsZWQsIGJ1dCBnb3QgYSAkeyR4LmR0eXBlfSB0ZW5zb3IgaW5zdGVhZC5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICByYXRlID49IDAgJiYgcmF0ZSA8IDEsXG4gICAgICAoKSA9PiBgcmF0ZSBtdXN0IGJlIGEgZmxvYXQgaW4gdGhlIHJhbmdlIFswLCAxKSwgYnV0IGdvdCAke3JhdGV9LmApO1xuXG4gIGlmIChyYXRlID09PSAwKSB7XG4gICAgcmV0dXJuIHggaW5zdGFuY2VvZiBUZW5zb3IgPyAkeC5jbG9uZSgpIDogJHg7XG4gIH1cblxuICBjb25zdCAkbm9pc2VTaGFwZSA9IGdldE5vaXNlU2hhcGUoJHgsIG5vaXNlU2hhcGUpO1xuICBjb25zdCBrZWVwUHJvYiA9IDEgLSByYXRlO1xuICBjb25zdCBtdWx0aXBsaWVyID0gZGl2KFxuICAgICAgZmxvb3IoYWRkKHJhbmRvbVVuaWZvcm0oJG5vaXNlU2hhcGUsIDAsIDEsICdmbG9hdDMyJywgc2VlZCksIGtlZXBQcm9iKSksXG4gICAgICBrZWVwUHJvYik7XG5cbiAgcmV0dXJuIG11bCgkeCwgbXVsdGlwbGllcik7XG59XG5cbmV4cG9ydCBjb25zdCBkcm9wb3V0ID0gLyogQF9fUFVSRV9fICovIG9wKHtkcm9wb3V0X30pO1xuIl19