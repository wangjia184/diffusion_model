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
import { Bincount } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Outputs a vector with length `size` and the same dtype as `weights`.
 *
 * If `weights` are empty, then index `i` stores the number of times the value
 * `i` is counted in `x`. If `weights` are non-empty, then index `i` stores the
 * sum of the value in `weights` at each index where the corresponding value in
 * `x` is `i`.
 *
 * Values in `x` outside of the range [0, size) are ignored.
 *
 * @param x The input int tensor, rank 1.
 * @param weights The weights tensor, must have the same shape as x, or a
 *     length-0 Tensor, in which case it acts as all weights equal to 1.
 * @param size Non-negative integer.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function bincount_(x, weights, size) {
    const $x = convertToTensor(x, 'x', 'bincount');
    const $weights = convertToTensor(weights, 'weights', 'bincount');
    util.assert($x.dtype === 'int32', () => `Error in bincount: input ` +
        `dtype must be int32, but got ${$x.dtype}`);
    util.assert(size >= 0, () => `size must be non-negative, but got ${size}.`);
    util.assert($weights.size === $x.size || $weights.size === 0, () => `Error in bincount: weights must have the same size as input or` +
        `0-length, but got input shape: ${$x.shape}, weights shape: ` +
        `${$weights.shape}.`);
    const inputs = { x: $x, weights: $weights };
    const attrs = { size };
    return ENGINE.runKernel(Bincount, inputs, attrs);
}
export const bincount = /* @__PURE__ */ op({ bincount_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmluY291bnQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9iaW5jb3VudC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxRQUFRLEVBQWdDLE1BQU0saUJBQWlCLENBQUM7QUFJeEUsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7R0FnQkc7QUFDSCxTQUFTLFNBQVMsQ0FDZCxDQUFlLEVBQUUsT0FBcUIsRUFBRSxJQUFZO0lBQ3RELE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFVBQVUsQ0FBQyxDQUFDO0lBQy9DLE1BQU0sUUFBUSxHQUFHLGVBQWUsQ0FBQyxPQUFPLEVBQUUsU0FBUyxFQUFFLFVBQVUsQ0FBQyxDQUFDO0lBRWpFLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLEtBQUssS0FBSyxPQUFPLEVBQ3BCLEdBQUcsRUFBRSxDQUFDLDJCQUEyQjtRQUM3QixnQ0FBZ0MsRUFBRSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7SUFDcEQsSUFBSSxDQUFDLE1BQU0sQ0FBQyxJQUFJLElBQUksQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLHNDQUFzQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBQzVFLElBQUksQ0FBQyxNQUFNLENBQ1AsUUFBUSxDQUFDLElBQUksS0FBSyxFQUFFLENBQUMsSUFBSSxJQUFJLFFBQVEsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNoRCxHQUFHLEVBQUUsQ0FBQyxnRUFBZ0U7UUFDbEUsa0NBQWtDLEVBQUUsQ0FBQyxLQUFLLG1CQUFtQjtRQUM3RCxHQUFHLFFBQVEsQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDO0lBRTlCLE1BQU0sTUFBTSxHQUFtQixFQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBQyxDQUFDO0lBQzFELE1BQU0sS0FBSyxHQUFrQixFQUFDLElBQUksRUFBQyxDQUFDO0lBRXBDLE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsUUFBUSxFQUFFLE1BQW1DLEVBQzdDLEtBQWdDLENBQUMsQ0FBQztBQUN4QyxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sUUFBUSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxTQUFTLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7QmluY291bnQsIEJpbmNvdW50QXR0cnMsIEJpbmNvdW50SW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjFEfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogT3V0cHV0cyBhIHZlY3RvciB3aXRoIGxlbmd0aCBgc2l6ZWAgYW5kIHRoZSBzYW1lIGR0eXBlIGFzIGB3ZWlnaHRzYC5cbiAqXG4gKiBJZiBgd2VpZ2h0c2AgYXJlIGVtcHR5LCB0aGVuIGluZGV4IGBpYCBzdG9yZXMgdGhlIG51bWJlciBvZiB0aW1lcyB0aGUgdmFsdWVcbiAqIGBpYCBpcyBjb3VudGVkIGluIGB4YC4gSWYgYHdlaWdodHNgIGFyZSBub24tZW1wdHksIHRoZW4gaW5kZXggYGlgIHN0b3JlcyB0aGVcbiAqIHN1bSBvZiB0aGUgdmFsdWUgaW4gYHdlaWdodHNgIGF0IGVhY2ggaW5kZXggd2hlcmUgdGhlIGNvcnJlc3BvbmRpbmcgdmFsdWUgaW5cbiAqIGB4YCBpcyBgaWAuXG4gKlxuICogVmFsdWVzIGluIGB4YCBvdXRzaWRlIG9mIHRoZSByYW5nZSBbMCwgc2l6ZSkgYXJlIGlnbm9yZWQuXG4gKlxuICogQHBhcmFtIHggVGhlIGlucHV0IGludCB0ZW5zb3IsIHJhbmsgMS5cbiAqIEBwYXJhbSB3ZWlnaHRzIFRoZSB3ZWlnaHRzIHRlbnNvciwgbXVzdCBoYXZlIHRoZSBzYW1lIHNoYXBlIGFzIHgsIG9yIGFcbiAqICAgICBsZW5ndGgtMCBUZW5zb3IsIGluIHdoaWNoIGNhc2UgaXQgYWN0cyBhcyBhbGwgd2VpZ2h0cyBlcXVhbCB0byAxLlxuICogQHBhcmFtIHNpemUgTm9uLW5lZ2F0aXZlIGludGVnZXIuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnUmVkdWN0aW9uJ31cbiAqL1xuZnVuY3Rpb24gYmluY291bnRfPFQgZXh0ZW5kcyBUZW5zb3IxRD4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCB3ZWlnaHRzOiBUfFRlbnNvckxpa2UsIHNpemU6IG51bWJlcik6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdiaW5jb3VudCcpO1xuICBjb25zdCAkd2VpZ2h0cyA9IGNvbnZlcnRUb1RlbnNvcih3ZWlnaHRzLCAnd2VpZ2h0cycsICdiaW5jb3VudCcpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHguZHR5cGUgPT09ICdpbnQzMicsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gYmluY291bnQ6IGlucHV0IGAgK1xuICAgICAgICAgIGBkdHlwZSBtdXN0IGJlIGludDMyLCBidXQgZ290ICR7JHguZHR5cGV9YCk7XG4gIHV0aWwuYXNzZXJ0KHNpemUgPj0gMCwgKCkgPT4gYHNpemUgbXVzdCBiZSBub24tbmVnYXRpdmUsIGJ1dCBnb3QgJHtzaXplfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkd2VpZ2h0cy5zaXplID09PSAkeC5zaXplIHx8ICR3ZWlnaHRzLnNpemUgPT09IDAsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gYmluY291bnQ6IHdlaWdodHMgbXVzdCBoYXZlIHRoZSBzYW1lIHNpemUgYXMgaW5wdXQgb3JgICtcbiAgICAgICAgICBgMC1sZW5ndGgsIGJ1dCBnb3QgaW5wdXQgc2hhcGU6ICR7JHguc2hhcGV9LCB3ZWlnaHRzIHNoYXBlOiBgICtcbiAgICAgICAgICBgJHskd2VpZ2h0cy5zaGFwZX0uYCk7XG5cbiAgY29uc3QgaW5wdXRzOiBCaW5jb3VudElucHV0cyA9IHt4OiAkeCwgd2VpZ2h0czogJHdlaWdodHN9O1xuICBjb25zdCBhdHRyczogQmluY291bnRBdHRycyA9IHtzaXplfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIEJpbmNvdW50LCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGJpbmNvdW50ID0gLyogQF9fUFVSRV9fICovIG9wKHtiaW5jb3VudF99KTtcbiJdfQ==