/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import { ENGINE } from '../../engine';
import { StringNGrams } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import { op } from '../operation';
/**
 * Creates ngrams from ragged string data.
 *
 * This op accepts a ragged tensor with 1 ragged dimension containing only
 * strings and outputs a ragged tensor with 1 ragged dimension containing ngrams
 * of that string, joined along the innermost axis.
 *
 * ```js
 * const result = tf.string.stringNGrams(
 *   ['a', 'b', 'c', 'd'], tf.tensor1d([0, 2, 4], 'int32'),
 *   '|', [1, 2], 'LP', 'RP', -1, false);
 * result['nGrams'].print(); // ['a', 'b', 'LP|a', 'a|b', 'b|RP',
 *                           //  'c', 'd', 'LP|c', 'c|d', 'd|RP']
 * result['nGramsSplits'].print(); // [0, 5, 10]
 * ```
 * @param data: The values tensor of the ragged string tensor to make ngrams out
 *     of. Must be a 1D string tensor.
 * @param dataSplits: The splits tensor of the ragged string tensor to make
 *     ngrams out of.
 * @param separator: The string to append between elements of the token. Use ""
 *     for no separator.
 * @param nGramWidths: The sizes of the ngrams to create.
 * @param leftPad: The string to use to pad the left side of the ngram sequence.
 *     Only used if pad_width !== 0.
 * @param rightPad: The string to use to pad the right side of the ngram
 *     sequence. Only used if pad_width !== 0.
 * @param padWidth: The number of padding elements to add to each side of each
 *     sequence. Note that padding will never be greater than `nGramWidths`-1
 *     regardless of this value. If `padWidth`=-1, then add max(`nGramWidths`)-1
 *     elements.
 * @param preserveShortSequences: If true, then ensure that at least one ngram
 *     is generated for each input sequence. In particular, if an input sequence
 *     is shorter than min(ngramWidth) + 2*padWidth, then generate a single
 *     ngram containing the entire sequence. If false, then no ngrams are
 *     generated for these short input sequences.
 * @return A map with the following properties:
 *     - nGrams: The values tensor of the output ngrams ragged tensor.
 *     - nGramsSplits: The splits tensor of the output ngrams ragged tensor.
 *
 * @doc {heading: 'Operations', subheading: 'String'}
 */
function stringNGrams_(data, dataSplits, separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences) {
    const $data = convertToTensor(data, 'data', 'stringNGrams', 'string');
    if ($data.dtype !== 'string') {
        throw new Error('Data must be of datatype string');
    }
    if ($data.shape.length !== 1) {
        throw new Error(`Data must be a vector, saw: ${$data.shape}`);
    }
    const $dataSplits = convertToTensor(dataSplits, 'dataSplits', 'stringNGrams');
    if ($dataSplits.dtype !== 'int32') {
        throw new Error('Data splits must be of datatype int32');
    }
    const attrs = {
        separator,
        nGramWidths,
        leftPad,
        rightPad,
        padWidth,
        preserveShortSequences
    };
    const inputs = { data: $data, dataSplits: $dataSplits };
    const result = ENGINE.runKernel(StringNGrams, inputs, attrs);
    return { nGrams: result[0], nGramsSplits: result[1] };
}
export const stringNGrams = /* @__PURE__ */ op({ stringNGrams_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3RyaW5nX25fZ3JhbXMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9zdHJpbmcvc3RyaW5nX25fZ3JhbXMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNwQyxPQUFPLEVBQUMsWUFBWSxFQUF3QyxNQUFNLG9CQUFvQixDQUFDO0FBR3ZGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV0RCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRWhDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBd0NHO0FBQ0gsU0FBUyxhQUFhLENBQ2xCLElBQXlCLEVBQUUsVUFBNkIsRUFBRSxTQUFpQixFQUMzRSxXQUFxQixFQUFFLE9BQWUsRUFBRSxRQUFnQixFQUFFLFFBQWdCLEVBQzFFLHNCQUErQjtJQUNqQyxNQUFNLEtBQUssR0FBRyxlQUFlLENBQUMsSUFBSSxFQUFFLE1BQU0sRUFBRSxjQUFjLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFDdEUsSUFBSSxLQUFLLENBQUMsS0FBSyxLQUFLLFFBQVEsRUFBRTtRQUM1QixNQUFNLElBQUksS0FBSyxDQUFDLGlDQUFpQyxDQUFDLENBQUM7S0FDcEQ7SUFDRCxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtRQUM1QixNQUFNLElBQUksS0FBSyxDQUFDLCtCQUErQixLQUFLLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUMvRDtJQUVELE1BQU0sV0FBVyxHQUFHLGVBQWUsQ0FBQyxVQUFVLEVBQUUsWUFBWSxFQUFFLGNBQWMsQ0FBQyxDQUFDO0lBQzlFLElBQUksV0FBVyxDQUFDLEtBQUssS0FBSyxPQUFPLEVBQUU7UUFDakMsTUFBTSxJQUFJLEtBQUssQ0FBQyx1Q0FBdUMsQ0FBQyxDQUFDO0tBQzFEO0lBRUQsTUFBTSxLQUFLLEdBQXNCO1FBQy9CLFNBQVM7UUFDVCxXQUFXO1FBQ1gsT0FBTztRQUNQLFFBQVE7UUFDUixRQUFRO1FBQ1Isc0JBQXNCO0tBQ3ZCLENBQUM7SUFFRixNQUFNLE1BQU0sR0FBdUIsRUFBQyxJQUFJLEVBQUUsS0FBSyxFQUFFLFVBQVUsRUFBRSxXQUFXLEVBQUMsQ0FBQztJQUMxRSxNQUFNLE1BQU0sR0FDUixNQUFNLENBQUMsU0FBUyxDQUFDLFlBQVksRUFBRSxNQUFZLEVBQUUsS0FBVyxDQUFDLENBQUM7SUFDOUQsT0FBTyxFQUFDLE1BQU0sRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUUsWUFBWSxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBQyxDQUFDO0FBQ3RELENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxZQUFZLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLGFBQWEsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtTdHJpbmdOR3JhbXMsIFN0cmluZ05HcmFtc0F0dHJzLCBTdHJpbmdOR3JhbXNJbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge1RlbnNvciwgVGVuc29yMUR9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi8uLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uLy4uL3R5cGVzJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5cbi8qKlxuICogQ3JlYXRlcyBuZ3JhbXMgZnJvbSByYWdnZWQgc3RyaW5nIGRhdGEuXG4gKlxuICogVGhpcyBvcCBhY2NlcHRzIGEgcmFnZ2VkIHRlbnNvciB3aXRoIDEgcmFnZ2VkIGRpbWVuc2lvbiBjb250YWluaW5nIG9ubHlcbiAqIHN0cmluZ3MgYW5kIG91dHB1dHMgYSByYWdnZWQgdGVuc29yIHdpdGggMSByYWdnZWQgZGltZW5zaW9uIGNvbnRhaW5pbmcgbmdyYW1zXG4gKiBvZiB0aGF0IHN0cmluZywgam9pbmVkIGFsb25nIHRoZSBpbm5lcm1vc3QgYXhpcy5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgcmVzdWx0ID0gdGYuc3RyaW5nLnN0cmluZ05HcmFtcyhcbiAqICAgWydhJywgJ2InLCAnYycsICdkJ10sIHRmLnRlbnNvcjFkKFswLCAyLCA0XSwgJ2ludDMyJyksXG4gKiAgICd8JywgWzEsIDJdLCAnTFAnLCAnUlAnLCAtMSwgZmFsc2UpO1xuICogcmVzdWx0WyduR3JhbXMnXS5wcmludCgpOyAvLyBbJ2EnLCAnYicsICdMUHxhJywgJ2F8YicsICdifFJQJyxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gICdjJywgJ2QnLCAnTFB8YycsICdjfGQnLCAnZHxSUCddXG4gKiByZXN1bHRbJ25HcmFtc1NwbGl0cyddLnByaW50KCk7IC8vIFswLCA1LCAxMF1cbiAqIGBgYFxuICogQHBhcmFtIGRhdGE6IFRoZSB2YWx1ZXMgdGVuc29yIG9mIHRoZSByYWdnZWQgc3RyaW5nIHRlbnNvciB0byBtYWtlIG5ncmFtcyBvdXRcbiAqICAgICBvZi4gTXVzdCBiZSBhIDFEIHN0cmluZyB0ZW5zb3IuXG4gKiBAcGFyYW0gZGF0YVNwbGl0czogVGhlIHNwbGl0cyB0ZW5zb3Igb2YgdGhlIHJhZ2dlZCBzdHJpbmcgdGVuc29yIHRvIG1ha2VcbiAqICAgICBuZ3JhbXMgb3V0IG9mLlxuICogQHBhcmFtIHNlcGFyYXRvcjogVGhlIHN0cmluZyB0byBhcHBlbmQgYmV0d2VlbiBlbGVtZW50cyBvZiB0aGUgdG9rZW4uIFVzZSBcIlwiXG4gKiAgICAgZm9yIG5vIHNlcGFyYXRvci5cbiAqIEBwYXJhbSBuR3JhbVdpZHRoczogVGhlIHNpemVzIG9mIHRoZSBuZ3JhbXMgdG8gY3JlYXRlLlxuICogQHBhcmFtIGxlZnRQYWQ6IFRoZSBzdHJpbmcgdG8gdXNlIHRvIHBhZCB0aGUgbGVmdCBzaWRlIG9mIHRoZSBuZ3JhbSBzZXF1ZW5jZS5cbiAqICAgICBPbmx5IHVzZWQgaWYgcGFkX3dpZHRoICE9PSAwLlxuICogQHBhcmFtIHJpZ2h0UGFkOiBUaGUgc3RyaW5nIHRvIHVzZSB0byBwYWQgdGhlIHJpZ2h0IHNpZGUgb2YgdGhlIG5ncmFtXG4gKiAgICAgc2VxdWVuY2UuIE9ubHkgdXNlZCBpZiBwYWRfd2lkdGggIT09IDAuXG4gKiBAcGFyYW0gcGFkV2lkdGg6IFRoZSBudW1iZXIgb2YgcGFkZGluZyBlbGVtZW50cyB0byBhZGQgdG8gZWFjaCBzaWRlIG9mIGVhY2hcbiAqICAgICBzZXF1ZW5jZS4gTm90ZSB0aGF0IHBhZGRpbmcgd2lsbCBuZXZlciBiZSBncmVhdGVyIHRoYW4gYG5HcmFtV2lkdGhzYC0xXG4gKiAgICAgcmVnYXJkbGVzcyBvZiB0aGlzIHZhbHVlLiBJZiBgcGFkV2lkdGhgPS0xLCB0aGVuIGFkZCBtYXgoYG5HcmFtV2lkdGhzYCktMVxuICogICAgIGVsZW1lbnRzLlxuICogQHBhcmFtIHByZXNlcnZlU2hvcnRTZXF1ZW5jZXM6IElmIHRydWUsIHRoZW4gZW5zdXJlIHRoYXQgYXQgbGVhc3Qgb25lIG5ncmFtXG4gKiAgICAgaXMgZ2VuZXJhdGVkIGZvciBlYWNoIGlucHV0IHNlcXVlbmNlLiBJbiBwYXJ0aWN1bGFyLCBpZiBhbiBpbnB1dCBzZXF1ZW5jZVxuICogICAgIGlzIHNob3J0ZXIgdGhhbiBtaW4obmdyYW1XaWR0aCkgKyAyKnBhZFdpZHRoLCB0aGVuIGdlbmVyYXRlIGEgc2luZ2xlXG4gKiAgICAgbmdyYW0gY29udGFpbmluZyB0aGUgZW50aXJlIHNlcXVlbmNlLiBJZiBmYWxzZSwgdGhlbiBubyBuZ3JhbXMgYXJlXG4gKiAgICAgZ2VuZXJhdGVkIGZvciB0aGVzZSBzaG9ydCBpbnB1dCBzZXF1ZW5jZXMuXG4gKiBAcmV0dXJuIEEgbWFwIHdpdGggdGhlIGZvbGxvd2luZyBwcm9wZXJ0aWVzOlxuICogICAgIC0gbkdyYW1zOiBUaGUgdmFsdWVzIHRlbnNvciBvZiB0aGUgb3V0cHV0IG5ncmFtcyByYWdnZWQgdGVuc29yLlxuICogICAgIC0gbkdyYW1zU3BsaXRzOiBUaGUgc3BsaXRzIHRlbnNvciBvZiB0aGUgb3V0cHV0IG5ncmFtcyByYWdnZWQgdGVuc29yLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ1N0cmluZyd9XG4gKi9cbmZ1bmN0aW9uIHN0cmluZ05HcmFtc18oXG4gICAgZGF0YTogVGVuc29yMUR8VGVuc29yTGlrZSwgZGF0YVNwbGl0czogVGVuc29yfFRlbnNvckxpa2UsIHNlcGFyYXRvcjogc3RyaW5nLFxuICAgIG5HcmFtV2lkdGhzOiBudW1iZXJbXSwgbGVmdFBhZDogc3RyaW5nLCByaWdodFBhZDogc3RyaW5nLCBwYWRXaWR0aDogbnVtYmVyLFxuICAgIHByZXNlcnZlU2hvcnRTZXF1ZW5jZXM6IGJvb2xlYW4pOiBOYW1lZFRlbnNvck1hcCB7XG4gIGNvbnN0ICRkYXRhID0gY29udmVydFRvVGVuc29yKGRhdGEsICdkYXRhJywgJ3N0cmluZ05HcmFtcycsICdzdHJpbmcnKTtcbiAgaWYgKCRkYXRhLmR0eXBlICE9PSAnc3RyaW5nJykge1xuICAgIHRocm93IG5ldyBFcnJvcignRGF0YSBtdXN0IGJlIG9mIGRhdGF0eXBlIHN0cmluZycpO1xuICB9XG4gIGlmICgkZGF0YS5zaGFwZS5sZW5ndGggIT09IDEpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYERhdGEgbXVzdCBiZSBhIHZlY3Rvciwgc2F3OiAkeyRkYXRhLnNoYXBlfWApO1xuICB9XG5cbiAgY29uc3QgJGRhdGFTcGxpdHMgPSBjb252ZXJ0VG9UZW5zb3IoZGF0YVNwbGl0cywgJ2RhdGFTcGxpdHMnLCAnc3RyaW5nTkdyYW1zJyk7XG4gIGlmICgkZGF0YVNwbGl0cy5kdHlwZSAhPT0gJ2ludDMyJykge1xuICAgIHRocm93IG5ldyBFcnJvcignRGF0YSBzcGxpdHMgbXVzdCBiZSBvZiBkYXRhdHlwZSBpbnQzMicpO1xuICB9XG5cbiAgY29uc3QgYXR0cnM6IFN0cmluZ05HcmFtc0F0dHJzID0ge1xuICAgIHNlcGFyYXRvcixcbiAgICBuR3JhbVdpZHRocyxcbiAgICBsZWZ0UGFkLFxuICAgIHJpZ2h0UGFkLFxuICAgIHBhZFdpZHRoLFxuICAgIHByZXNlcnZlU2hvcnRTZXF1ZW5jZXNcbiAgfTtcblxuICBjb25zdCBpbnB1dHM6IFN0cmluZ05HcmFtc0lucHV0cyA9IHtkYXRhOiAkZGF0YSwgZGF0YVNwbGl0czogJGRhdGFTcGxpdHN9O1xuICBjb25zdCByZXN1bHQ6IFRlbnNvcltdID1cbiAgICAgIEVOR0lORS5ydW5LZXJuZWwoU3RyaW5nTkdyYW1zLCBpbnB1dHMgYXMge30sIGF0dHJzIGFzIHt9KTtcbiAgcmV0dXJuIHtuR3JhbXM6IHJlc3VsdFswXSwgbkdyYW1zU3BsaXRzOiByZXN1bHRbMV19O1xufVxuXG5leHBvcnQgY29uc3Qgc3RyaW5nTkdyYW1zID0gLyogQF9fUFVSRV9fICovIG9wKHtzdHJpbmdOR3JhbXNffSk7XG4iXX0=