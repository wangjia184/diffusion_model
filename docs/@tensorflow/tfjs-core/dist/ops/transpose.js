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
import { ENGINE } from '../engine';
import { tidy } from '../globals';
import { Transpose } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { complex } from './complex';
import { imag } from './imag';
import { neg } from './neg';
import { op } from './operation';
import { real } from './real';
/**
 * Transposes the `tf.Tensor`. Permutes the dimensions according to `perm`.
 *
 * The returned `tf.Tensor`'s dimension `i` will correspond to the input
 * dimension `perm[i]`. If `perm` is not given, it is set to `[n-1...0]`,
 * where `n` is the rank of the input `tf.Tensor`. Hence by default, this
 * operation performs a regular matrix transpose on 2-D input `tf.Tensor`s.
 *
 * ```js
 * const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
 *
 * a.transpose().print();  // or tf.transpose(a)
 * ```
 *
 * @param x The tensor to transpose.
 * @param perm The permutation of the dimensions of a.
 * @param conjugate Will conjugate complex input if true.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function transpose_(x, perm, conjugate) {
    const $x = convertToTensor(x, 'x', 'transpose');
    if (perm == null) {
        perm = $x.shape.map((s, i) => i).reverse();
    }
    util.assert($x.rank === perm.length, () => `Error in transpose: rank of input ${$x.rank} ` +
        `must match length of perm ${perm}.`);
    perm.forEach(axis => {
        util.assert(axis >= 0 && axis < $x.rank, () => `All entries in 'perm' must be between 0 and ${$x.rank - 1}` +
            ` but got ${perm}`);
    });
    if ($x.rank <= 1) {
        return $x.clone();
    }
    const inputs = { x: $x };
    const attrs = { perm };
    if ($x.dtype === 'complex64') {
        return tidy(() => {
            let $real = real($x);
            let $imag = imag($x);
            $real = ENGINE.runKernel(Transpose, { x: $real }, attrs);
            $imag = ENGINE.runKernel(Transpose, { x: $imag }, attrs);
            if (conjugate) {
                $imag = neg($imag);
            }
            return complex($real, $imag);
        });
    }
    return ENGINE.runKernel(Transpose, inputs, attrs);
}
export const transpose = /* @__PURE__ */ op({ transpose_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhbnNwb3NlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvdHJhbnNwb3NlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNoQyxPQUFPLEVBQUMsU0FBUyxFQUFrQyxNQUFNLGlCQUFpQixDQUFDO0FBSTNFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2xDLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDNUIsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFFNUI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FtQkc7QUFDSCxTQUFTLFVBQVUsQ0FDZixDQUFlLEVBQUUsSUFBZSxFQUFFLFNBQW1CO0lBQ3ZELE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBRWhELElBQUksSUFBSSxJQUFJLElBQUksRUFBRTtRQUNoQixJQUFJLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztLQUM1QztJQUNELElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxJQUFJLENBQUMsTUFBTSxFQUN2QixHQUFHLEVBQUUsQ0FBQyxxQ0FBcUMsRUFBRSxDQUFDLElBQUksR0FBRztRQUNqRCw2QkFBNkIsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUM5QyxJQUFJLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxFQUFFO1FBQ2xCLElBQUksQ0FBQyxNQUFNLENBQ1AsSUFBSSxJQUFJLENBQUMsSUFBSSxJQUFJLEdBQUcsRUFBRSxDQUFDLElBQUksRUFDM0IsR0FBRyxFQUFFLENBQUMsK0NBQStDLEVBQUUsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxFQUFFO1lBQzlELFlBQVksSUFBSSxFQUFFLENBQUMsQ0FBQztJQUM5QixDQUFDLENBQUMsQ0FBQztJQUVILElBQUksRUFBRSxDQUFDLElBQUksSUFBSSxDQUFDLEVBQUU7UUFDaEIsT0FBTyxFQUFFLENBQUMsS0FBSyxFQUFFLENBQUM7S0FDbkI7SUFFRCxNQUFNLE1BQU0sR0FBb0IsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDeEMsTUFBTSxLQUFLLEdBQW1CLEVBQUMsSUFBSSxFQUFDLENBQUM7SUFFckMsSUFBSSxFQUFFLENBQUMsS0FBSyxLQUFLLFdBQVcsRUFBRTtRQUM1QixPQUFPLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDZixJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUM7WUFDckIsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1lBQ3JCLEtBQUssR0FBRyxNQUFNLENBQUMsU0FBUyxDQUNwQixTQUFTLEVBQUUsRUFBQyxDQUFDLEVBQUUsS0FBSyxFQUE4QixFQUNsRCxLQUFnQyxDQUFDLENBQUM7WUFDdEMsS0FBSyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ3BCLFNBQVMsRUFBRSxFQUFDLENBQUMsRUFBRSxLQUFLLEVBQThCLEVBQ2xELEtBQWdDLENBQUMsQ0FBQztZQUN0QyxJQUFJLFNBQVMsRUFBRTtnQkFDYixLQUFLLEdBQUcsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDO2FBQ3BCO1lBQ0QsT0FBTyxPQUFPLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDO1FBQy9CLENBQUMsQ0FBQyxDQUFDO0tBQ0o7SUFFRCxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLFNBQVMsRUFBRSxNQUFtQyxFQUM5QyxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFNBQVMsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsVUFBVSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge3RpZHl9IGZyb20gJy4uL2dsb2JhbHMnO1xuaW1wb3J0IHtUcmFuc3Bvc2UsIFRyYW5zcG9zZUF0dHJzLCBUcmFuc3Bvc2VJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuaW1wb3J0IHtjb21wbGV4fSBmcm9tICcuL2NvbXBsZXgnO1xuaW1wb3J0IHtpbWFnfSBmcm9tICcuL2ltYWcnO1xuaW1wb3J0IHtuZWd9IGZyb20gJy4vbmVnJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVhbH0gZnJvbSAnLi9yZWFsJztcblxuLyoqXG4gKiBUcmFuc3Bvc2VzIHRoZSBgdGYuVGVuc29yYC4gUGVybXV0ZXMgdGhlIGRpbWVuc2lvbnMgYWNjb3JkaW5nIHRvIGBwZXJtYC5cbiAqXG4gKiBUaGUgcmV0dXJuZWQgYHRmLlRlbnNvcmAncyBkaW1lbnNpb24gYGlgIHdpbGwgY29ycmVzcG9uZCB0byB0aGUgaW5wdXRcbiAqIGRpbWVuc2lvbiBgcGVybVtpXWAuIElmIGBwZXJtYCBpcyBub3QgZ2l2ZW4sIGl0IGlzIHNldCB0byBgW24tMS4uLjBdYCxcbiAqIHdoZXJlIGBuYCBpcyB0aGUgcmFuayBvZiB0aGUgaW5wdXQgYHRmLlRlbnNvcmAuIEhlbmNlIGJ5IGRlZmF1bHQsIHRoaXNcbiAqIG9wZXJhdGlvbiBwZXJmb3JtcyBhIHJlZ3VsYXIgbWF0cml4IHRyYW5zcG9zZSBvbiAyLUQgaW5wdXQgYHRmLlRlbnNvcmBzLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBhID0gdGYudGVuc29yMmQoWzEsIDIsIDMsIDQsIDUsIDZdLCBbMiwgM10pO1xuICpcbiAqIGEudHJhbnNwb3NlKCkucHJpbnQoKTsgIC8vIG9yIHRmLnRyYW5zcG9zZShhKVxuICogYGBgXG4gKlxuICogQHBhcmFtIHggVGhlIHRlbnNvciB0byB0cmFuc3Bvc2UuXG4gKiBAcGFyYW0gcGVybSBUaGUgcGVybXV0YXRpb24gb2YgdGhlIGRpbWVuc2lvbnMgb2YgYS5cbiAqIEBwYXJhbSBjb25qdWdhdGUgV2lsbCBjb25qdWdhdGUgY29tcGxleCBpbnB1dCBpZiB0cnVlLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ01hdHJpY2VzJ31cbiAqL1xuZnVuY3Rpb24gdHJhbnNwb3NlXzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICB4OiBUfFRlbnNvckxpa2UsIHBlcm0/OiBudW1iZXJbXSwgY29uanVnYXRlPzogYm9vbGVhbik6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICd0cmFuc3Bvc2UnKTtcblxuICBpZiAocGVybSA9PSBudWxsKSB7XG4gICAgcGVybSA9ICR4LnNoYXBlLm1hcCgocywgaSkgPT4gaSkucmV2ZXJzZSgpO1xuICB9XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHgucmFuayA9PT0gcGVybS5sZW5ndGgsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gdHJhbnNwb3NlOiByYW5rIG9mIGlucHV0ICR7JHgucmFua30gYCArXG4gICAgICAgICAgYG11c3QgbWF0Y2ggbGVuZ3RoIG9mIHBlcm0gJHtwZXJtfS5gKTtcbiAgcGVybS5mb3JFYWNoKGF4aXMgPT4ge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBheGlzID49IDAgJiYgYXhpcyA8ICR4LnJhbmssXG4gICAgICAgICgpID0+IGBBbGwgZW50cmllcyBpbiAncGVybScgbXVzdCBiZSBiZXR3ZWVuIDAgYW5kICR7JHgucmFuayAtIDF9YCArXG4gICAgICAgICAgICBgIGJ1dCBnb3QgJHtwZXJtfWApO1xuICB9KTtcblxuICBpZiAoJHgucmFuayA8PSAxKSB7XG4gICAgcmV0dXJuICR4LmNsb25lKCk7XG4gIH1cblxuICBjb25zdCBpbnB1dHM6IFRyYW5zcG9zZUlucHV0cyA9IHt4OiAkeH07XG4gIGNvbnN0IGF0dHJzOiBUcmFuc3Bvc2VBdHRycyA9IHtwZXJtfTtcblxuICBpZiAoJHguZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgbGV0ICRyZWFsID0gcmVhbCgkeCk7XG4gICAgICBsZXQgJGltYWcgPSBpbWFnKCR4KTtcbiAgICAgICRyZWFsID0gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgICAgICBUcmFuc3Bvc2UsIHt4OiAkcmVhbH0gYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG4gICAgICAkaW1hZyA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgVHJhbnNwb3NlLCB7eDogJGltYWd9IGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xuICAgICAgaWYgKGNvbmp1Z2F0ZSkge1xuICAgICAgICAkaW1hZyA9IG5lZygkaW1hZyk7XG4gICAgICB9XG4gICAgICByZXR1cm4gY29tcGxleCgkcmVhbCwgJGltYWcpO1xuICAgIH0pO1xuICB9XG5cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBUcmFuc3Bvc2UsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3QgdHJhbnNwb3NlID0gLyogQF9fUFVSRV9fICovIG9wKHt0cmFuc3Bvc2VffSk7XG4iXX0=