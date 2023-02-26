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
import { MirrorPad } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Pads a `tf.Tensor` using mirror padding.
 *
 * This operation implements the `REFLECT` and `SYMMETRIC` modes of pad.
 *
 * ```js
 * const x = tf.range(0, 9).reshape([1, 1, 3, 3]);
 * x.mirrorPad([[0, 0], [0, 0], [2, 2], [2, 2]], 'reflect').print();
 * ```
 * @param x The tensor to pad.
 * @param paddings An array of length `R` (the rank of the tensor), where
 * each element is a length-2 tuple of ints `[padBefore, padAfter]`,
 * specifying how much to pad along each dimension of the tensor.
 * In "reflect" mode, the padded regions do not include the borders,
 * while in "symmetric" mode the padded regions do include the borders.
 * For example, if the input is `[1, 2, 3]` and paddings is `[0, 2]`,
 * then the output is `[1, 2, 3, 2, 1]` in "reflect" mode, and
 * `[1, 2, 3, 3, 2]` in "symmetric" mode.
 * If `mode` is "reflect" then both `paddings[D, 0]` and `paddings[D, 1]`
 * must be no greater than `x.shape[D] - 1`. If mode is "symmetric"
 * then both `paddings[D, 0]` and `paddings[D, 1]` must be no greater than
 * `x.shape[D]`
 * @param mode String to specify padding mode. Can be `'reflect' | 'symmetric'`
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function mirrorPad_(x, paddings, mode) {
    util.assert(mode === 'reflect' || mode === 'symmetric', () => `Invalid mode. Mode must be either reflect or symmetric. ` +
        `Got ${mode}.`);
    const $x = convertToTensor(x, 'x', 'mirrorPad');
    if ($x.rank === 0) {
        throw new Error('mirrorPad(scalar) is not defined. ' +
            'Pass non-scalar to mirrorPad');
    }
    util.assert(paddings.length === $x.rank, () => `Padding doesn't match input. Must be ${$x.rank}. ` +
        `Got ${paddings.length}.`);
    const shapeOffset = mode === 'reflect' ? 1 : 0;
    for (let i = 0; i < $x.rank; i++) {
        util.assert(paddings[i].length === 2, () => `Invalid number of paddings. Must be length of 2 each.`);
        util.assert(paddings[i][0] >= 0 && paddings[i][0] <= $x.shape[i] - shapeOffset &&
            paddings[i][1] >= 0 && paddings[i][1] <= $x.shape[i] - shapeOffset, () => `Padding in dimension ${i} cannot be greater than or equal ` +
            `to ${$x.shape[i] - shapeOffset} or less than 0 for input of ` +
            `shape ${$x.shape}`);
    }
    const attrs = { paddings, mode };
    const inputs = { x: $x };
    return ENGINE.runKernel(MirrorPad, inputs, attrs);
}
export const mirrorPad = /* @__PURE__ */ op({ mirrorPad_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWlycm9yX3BhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL21pcnJvcl9wYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsU0FBUyxFQUFrQyxNQUFNLGlCQUFpQixDQUFDO0FBSTNFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXVCRztBQUNILCtEQUErRDtBQUMvRCxTQUFTLFVBQVUsQ0FDZixDQUFlLEVBQUUsUUFBaUMsRUFDbEQsSUFBMkI7SUFDN0IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLEtBQUssU0FBUyxJQUFJLElBQUksS0FBSyxXQUFXLEVBQzFDLEdBQUcsRUFBRSxDQUFDLDBEQUEwRDtRQUM1RCxPQUFPLElBQUksR0FBRyxDQUFDLENBQUM7SUFFeEIsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFDaEQsSUFBSSxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNqQixNQUFNLElBQUksS0FBSyxDQUNYLG9DQUFvQztZQUNwQyw4QkFBOEIsQ0FBQyxDQUFDO0tBQ3JDO0lBQ0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLENBQUMsTUFBTSxLQUFLLEVBQUUsQ0FBQyxJQUFJLEVBQzNCLEdBQUcsRUFBRSxDQUFDLHdDQUF3QyxFQUFFLENBQUMsSUFBSSxJQUFJO1FBQ3JELE9BQU8sUUFBUSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUM7SUFDbkMsTUFBTSxXQUFXLEdBQUcsSUFBSSxLQUFLLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDL0MsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxFQUFFLEVBQUU7UUFDaEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDeEIsR0FBRyxFQUFFLENBQUMsdURBQXVELENBQUMsQ0FBQztRQUNuRSxJQUFJLENBQUMsTUFBTSxDQUNQLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsV0FBVztZQUM5RCxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLFdBQVcsRUFDdEUsR0FBRyxFQUFFLENBQUMsd0JBQXdCLENBQUMsbUNBQW1DO1lBQzlELE1BQU0sRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxXQUFXLCtCQUErQjtZQUM5RCxTQUFTLEVBQUUsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzlCO0lBRUQsTUFBTSxLQUFLLEdBQW1CLEVBQUMsUUFBUSxFQUFFLElBQUksRUFBQyxDQUFDO0lBQy9DLE1BQU0sTUFBTSxHQUFvQixFQUFDLENBQUMsRUFBRSxFQUFFLEVBQUMsQ0FBQztJQUN4QyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLFNBQVMsRUFBRSxNQUFtQyxFQUM5QyxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFNBQVMsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsVUFBVSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge01pcnJvclBhZCwgTWlycm9yUGFkQXR0cnMsIE1pcnJvclBhZElucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBQYWRzIGEgYHRmLlRlbnNvcmAgdXNpbmcgbWlycm9yIHBhZGRpbmcuXG4gKlxuICogVGhpcyBvcGVyYXRpb24gaW1wbGVtZW50cyB0aGUgYFJFRkxFQ1RgIGFuZCBgU1lNTUVUUklDYCBtb2RlcyBvZiBwYWQuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi5yYW5nZSgwLCA5KS5yZXNoYXBlKFsxLCAxLCAzLCAzXSk7XG4gKiB4Lm1pcnJvclBhZChbWzAsIDBdLCBbMCwgMF0sIFsyLCAyXSwgWzIsIDJdXSwgJ3JlZmxlY3QnKS5wcmludCgpO1xuICogYGBgXG4gKiBAcGFyYW0geCBUaGUgdGVuc29yIHRvIHBhZC5cbiAqIEBwYXJhbSBwYWRkaW5ncyBBbiBhcnJheSBvZiBsZW5ndGggYFJgICh0aGUgcmFuayBvZiB0aGUgdGVuc29yKSwgd2hlcmVcbiAqIGVhY2ggZWxlbWVudCBpcyBhIGxlbmd0aC0yIHR1cGxlIG9mIGludHMgYFtwYWRCZWZvcmUsIHBhZEFmdGVyXWAsXG4gKiBzcGVjaWZ5aW5nIGhvdyBtdWNoIHRvIHBhZCBhbG9uZyBlYWNoIGRpbWVuc2lvbiBvZiB0aGUgdGVuc29yLlxuICogSW4gXCJyZWZsZWN0XCIgbW9kZSwgdGhlIHBhZGRlZCByZWdpb25zIGRvIG5vdCBpbmNsdWRlIHRoZSBib3JkZXJzLFxuICogd2hpbGUgaW4gXCJzeW1tZXRyaWNcIiBtb2RlIHRoZSBwYWRkZWQgcmVnaW9ucyBkbyBpbmNsdWRlIHRoZSBib3JkZXJzLlxuICogRm9yIGV4YW1wbGUsIGlmIHRoZSBpbnB1dCBpcyBgWzEsIDIsIDNdYCBhbmQgcGFkZGluZ3MgaXMgYFswLCAyXWAsXG4gKiB0aGVuIHRoZSBvdXRwdXQgaXMgYFsxLCAyLCAzLCAyLCAxXWAgaW4gXCJyZWZsZWN0XCIgbW9kZSwgYW5kXG4gKiBgWzEsIDIsIDMsIDMsIDJdYCBpbiBcInN5bW1ldHJpY1wiIG1vZGUuXG4gKiBJZiBgbW9kZWAgaXMgXCJyZWZsZWN0XCIgdGhlbiBib3RoIGBwYWRkaW5nc1tELCAwXWAgYW5kIGBwYWRkaW5nc1tELCAxXWBcbiAqIG11c3QgYmUgbm8gZ3JlYXRlciB0aGFuIGB4LnNoYXBlW0RdIC0gMWAuIElmIG1vZGUgaXMgXCJzeW1tZXRyaWNcIlxuICogdGhlbiBib3RoIGBwYWRkaW5nc1tELCAwXWAgYW5kIGBwYWRkaW5nc1tELCAxXWAgbXVzdCBiZSBubyBncmVhdGVyIHRoYW5cbiAqIGB4LnNoYXBlW0RdYFxuICogQHBhcmFtIG1vZGUgU3RyaW5nIHRvIHNwZWNpZnkgcGFkZGluZyBtb2RlLiBDYW4gYmUgYCdyZWZsZWN0JyB8ICdzeW1tZXRyaWMnYFxuICovXG4vKiogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnVHJhbnNmb3JtYXRpb25zJ30gKi9cbmZ1bmN0aW9uIG1pcnJvclBhZF88VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCBwYWRkaW5nczogQXJyYXk8W251bWJlciwgbnVtYmVyXT4sXG4gICAgbW9kZTogJ3JlZmxlY3QnfCdzeW1tZXRyaWMnKTogVCB7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgbW9kZSA9PT0gJ3JlZmxlY3QnIHx8IG1vZGUgPT09ICdzeW1tZXRyaWMnLFxuICAgICAgKCkgPT4gYEludmFsaWQgbW9kZS4gTW9kZSBtdXN0IGJlIGVpdGhlciByZWZsZWN0IG9yIHN5bW1ldHJpYy4gYCArXG4gICAgICAgICAgYEdvdCAke21vZGV9LmApO1xuXG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ21pcnJvclBhZCcpO1xuICBpZiAoJHgucmFuayA9PT0gMCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ21pcnJvclBhZChzY2FsYXIpIGlzIG5vdCBkZWZpbmVkLiAnICtcbiAgICAgICAgJ1Bhc3Mgbm9uLXNjYWxhciB0byBtaXJyb3JQYWQnKTtcbiAgfVxuICB1dGlsLmFzc2VydChcbiAgICAgIHBhZGRpbmdzLmxlbmd0aCA9PT0gJHgucmFuayxcbiAgICAgICgpID0+IGBQYWRkaW5nIGRvZXNuJ3QgbWF0Y2ggaW5wdXQuIE11c3QgYmUgJHskeC5yYW5rfS4gYCArXG4gICAgICAgICAgYEdvdCAke3BhZGRpbmdzLmxlbmd0aH0uYCk7XG4gIGNvbnN0IHNoYXBlT2Zmc2V0ID0gbW9kZSA9PT0gJ3JlZmxlY3QnID8gMSA6IDA7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgJHgucmFuazsgaSsrKSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHBhZGRpbmdzW2ldLmxlbmd0aCA9PT0gMixcbiAgICAgICAgKCkgPT4gYEludmFsaWQgbnVtYmVyIG9mIHBhZGRpbmdzLiBNdXN0IGJlIGxlbmd0aCBvZiAyIGVhY2guYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHBhZGRpbmdzW2ldWzBdID49IDAgJiYgcGFkZGluZ3NbaV1bMF0gPD0gJHguc2hhcGVbaV0gLSBzaGFwZU9mZnNldCAmJlxuICAgICAgICAgICAgcGFkZGluZ3NbaV1bMV0gPj0gMCAmJiBwYWRkaW5nc1tpXVsxXSA8PSAkeC5zaGFwZVtpXSAtIHNoYXBlT2Zmc2V0LFxuICAgICAgICAoKSA9PiBgUGFkZGluZyBpbiBkaW1lbnNpb24gJHtpfSBjYW5ub3QgYmUgZ3JlYXRlciB0aGFuIG9yIGVxdWFsIGAgK1xuICAgICAgICAgICAgYHRvICR7JHguc2hhcGVbaV0gLSBzaGFwZU9mZnNldH0gb3IgbGVzcyB0aGFuIDAgZm9yIGlucHV0IG9mIGAgK1xuICAgICAgICAgICAgYHNoYXBlICR7JHguc2hhcGV9YCk7XG4gIH1cblxuICBjb25zdCBhdHRyczogTWlycm9yUGFkQXR0cnMgPSB7cGFkZGluZ3MsIG1vZGV9O1xuICBjb25zdCBpbnB1dHM6IE1pcnJvclBhZElucHV0cyA9IHt4OiAkeH07XG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgTWlycm9yUGFkLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IG1pcnJvclBhZCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7bWlycm9yUGFkX30pO1xuIl19