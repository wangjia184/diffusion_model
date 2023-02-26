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
import { StringToHashBucketFast } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import { op } from '../operation';
/**
 * Converts each string in the input Tensor to its hash mod by a number of
 * buckets.
 *
 * The hash function is deterministic on the content of the string within the
 * process and will never change. However, it is not suitable for cryptography.
 * This function may be used when CPU time is scarce and inputs are trusted or
 * unimportant. There is a risk of adversaries constructing inputs that all hash
 * to the same bucket.
 *
 * ```js
 * const result = tf.string.stringToHashBucketFast(
 *   ['Hello', 'TensorFlow', '2.x'], 3);
 * result.print(); // [0, 2, 2]
 * ```
 * @param input: The strings to assign a hash bucket.
 * @param numBuckets: The number of buckets.
 * @return A Tensor of the same shape as the input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'String'}
 */
function stringToHashBucketFast_(input, numBuckets) {
    const $input = convertToTensor(input, 'input', 'stringToHashBucketFast', 'string');
    const attrs = { numBuckets };
    if (numBuckets <= 0) {
        throw new Error(`Number of buckets must be at least 1`);
    }
    const inputs = { input: $input };
    return ENGINE.runKernel(StringToHashBucketFast, inputs, attrs);
}
export const stringToHashBucketFast = /* @__PURE__ */ op({ stringToHashBucketFast_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3RyaW5nX3RvX2hhc2hfYnVja2V0X2Zhc3QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9zdHJpbmcvc3RyaW5nX3RvX2hhc2hfYnVja2V0X2Zhc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNwQyxPQUFPLEVBQUMsc0JBQXNCLEVBQTRELE1BQU0sb0JBQW9CLENBQUM7QUFFckgsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXRELE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFFaEM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBb0JHO0FBQ0gsU0FBUyx1QkFBdUIsQ0FDNUIsS0FBd0IsRUFBRSxVQUFrQjtJQUM5QyxNQUFNLE1BQU0sR0FDUixlQUFlLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSx3QkFBd0IsRUFBRSxRQUFRLENBQUMsQ0FBQztJQUN4RSxNQUFNLEtBQUssR0FBZ0MsRUFBQyxVQUFVLEVBQUMsQ0FBQztJQUV4RCxJQUFJLFVBQVUsSUFBSSxDQUFDLEVBQUU7UUFDbkIsTUFBTSxJQUFJLEtBQUssQ0FBQyxzQ0FBc0MsQ0FBQyxDQUFDO0tBQ3pEO0lBRUQsTUFBTSxNQUFNLEdBQWlDLEVBQUMsS0FBSyxFQUFFLE1BQU0sRUFBQyxDQUFDO0lBQzdELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FBQyxzQkFBc0IsRUFBRSxNQUFZLEVBQUUsS0FBVyxDQUFDLENBQUM7QUFDN0UsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLHNCQUFzQixHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyx1QkFBdUIsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtTdHJpbmdUb0hhc2hCdWNrZXRGYXN0LCBTdHJpbmdUb0hhc2hCdWNrZXRGYXN0QXR0cnMsIFN0cmluZ1RvSGFzaEJ1Y2tldEZhc3RJbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi8uLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi8uLi90eXBlcyc7XG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIENvbnZlcnRzIGVhY2ggc3RyaW5nIGluIHRoZSBpbnB1dCBUZW5zb3IgdG8gaXRzIGhhc2ggbW9kIGJ5IGEgbnVtYmVyIG9mXG4gKiBidWNrZXRzLlxuICpcbiAqIFRoZSBoYXNoIGZ1bmN0aW9uIGlzIGRldGVybWluaXN0aWMgb24gdGhlIGNvbnRlbnQgb2YgdGhlIHN0cmluZyB3aXRoaW4gdGhlXG4gKiBwcm9jZXNzIGFuZCB3aWxsIG5ldmVyIGNoYW5nZS4gSG93ZXZlciwgaXQgaXMgbm90IHN1aXRhYmxlIGZvciBjcnlwdG9ncmFwaHkuXG4gKiBUaGlzIGZ1bmN0aW9uIG1heSBiZSB1c2VkIHdoZW4gQ1BVIHRpbWUgaXMgc2NhcmNlIGFuZCBpbnB1dHMgYXJlIHRydXN0ZWQgb3JcbiAqIHVuaW1wb3J0YW50LiBUaGVyZSBpcyBhIHJpc2sgb2YgYWR2ZXJzYXJpZXMgY29uc3RydWN0aW5nIGlucHV0cyB0aGF0IGFsbCBoYXNoXG4gKiB0byB0aGUgc2FtZSBidWNrZXQuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHJlc3VsdCA9IHRmLnN0cmluZy5zdHJpbmdUb0hhc2hCdWNrZXRGYXN0KFxuICogICBbJ0hlbGxvJywgJ1RlbnNvckZsb3cnLCAnMi54J10sIDMpO1xuICogcmVzdWx0LnByaW50KCk7IC8vIFswLCAyLCAyXVxuICogYGBgXG4gKiBAcGFyYW0gaW5wdXQ6IFRoZSBzdHJpbmdzIHRvIGFzc2lnbiBhIGhhc2ggYnVja2V0LlxuICogQHBhcmFtIG51bUJ1Y2tldHM6IFRoZSBudW1iZXIgb2YgYnVja2V0cy5cbiAqIEByZXR1cm4gQSBUZW5zb3Igb2YgdGhlIHNhbWUgc2hhcGUgYXMgdGhlIGlucHV0IHRlbnNvci5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdTdHJpbmcnfVxuICovXG5mdW5jdGlvbiBzdHJpbmdUb0hhc2hCdWNrZXRGYXN0XyhcbiAgICBpbnB1dDogVGVuc29yfFRlbnNvckxpa2UsIG51bUJ1Y2tldHM6IG51bWJlcik6IFRlbnNvciB7XG4gIGNvbnN0ICRpbnB1dCA9XG4gICAgICBjb252ZXJ0VG9UZW5zb3IoaW5wdXQsICdpbnB1dCcsICdzdHJpbmdUb0hhc2hCdWNrZXRGYXN0JywgJ3N0cmluZycpO1xuICBjb25zdCBhdHRyczogU3RyaW5nVG9IYXNoQnVja2V0RmFzdEF0dHJzID0ge251bUJ1Y2tldHN9O1xuXG4gIGlmIChudW1CdWNrZXRzIDw9IDApIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYE51bWJlciBvZiBidWNrZXRzIG11c3QgYmUgYXQgbGVhc3QgMWApO1xuICB9XG5cbiAgY29uc3QgaW5wdXRzOiBTdHJpbmdUb0hhc2hCdWNrZXRGYXN0SW5wdXRzID0ge2lucHV0OiAkaW5wdXR9O1xuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChTdHJpbmdUb0hhc2hCdWNrZXRGYXN0LCBpbnB1dHMgYXMge30sIGF0dHJzIGFzIHt9KTtcbn1cblxuZXhwb3J0IGNvbnN0IHN0cmluZ1RvSGFzaEJ1Y2tldEZhc3QgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3N0cmluZ1RvSGFzaEJ1Y2tldEZhc3RffSk7XG4iXX0=