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
import * as tf from '../index';
import { ALL_ENVS, describeWithFlags } from '../jasmine_util';
import { expectArrayInMeanStdRange } from './rand_util';
describeWithFlags('truncatedNormal', ALL_ENVS, () => {
    // Expect slightly higher variances for truncated values.
    const EPSILON = 0.60;
    const SEED = 2002;
    function assertTruncatedValues(values, mean, stdv) {
        const bounds = mean + stdv * 2;
        for (let i = 0; i < values.length; i++) {
            expect(Math.abs(values[i])).toBeLessThanOrEqual(bounds);
        }
    }
    it('should return a random 1D float32 array', async () => {
        const shape = [1000];
        // Ensure defaults to float32 w/o type:
        let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(await result.data(), 0, 3.5);
        expectArrayInMeanStdRange(await result.data(), 0, 3.5, EPSILON);
        result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(await result.data(), 0, 4.5);
        expectArrayInMeanStdRange(await result.data(), 0, 4.5, EPSILON);
    });
    it('should return a randon 1D int32 array', async () => {
        const shape = [1000];
        const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(await result.data(), 0, 5);
        expectArrayInMeanStdRange(await result.data(), 0, 5, EPSILON);
    });
    it('should return a 2D float32 array', async () => {
        const shape = [50, 50];
        // Ensure defaults to float32 w/o type:
        let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(await result.data(), 0, 3.5);
        expectArrayInMeanStdRange(await result.data(), 0, 3.5, EPSILON);
        result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(await result.data(), 0, 4.5);
        expectArrayInMeanStdRange(await result.data(), 0, 4.5, EPSILON);
    });
    it('should return a 2D int32 array', async () => {
        const shape = [50, 50];
        const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(await result.data(), 0, 5);
        expectArrayInMeanStdRange(await result.data(), 0, 5, EPSILON);
    });
    it('should return a 3D float32 array', async () => {
        const shape = [10, 10, 10];
        // Ensure defaults to float32 w/o type:
        let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(await result.data(), 0, 3.5);
        expectArrayInMeanStdRange(await result.data(), 0, 3.5, EPSILON);
        result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(await result.data(), 0, 4.5);
        expectArrayInMeanStdRange(await result.data(), 0, 4.5, EPSILON);
    });
    it('should return a 3D int32 array', async () => {
        const shape = [10, 10, 10];
        const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(await result.data(), 0, 5);
        expectArrayInMeanStdRange(await result.data(), 0, 5, EPSILON);
    });
    it('should return a 4D float32 array', async () => {
        const shape = [5, 5, 5, 5];
        // Ensure defaults to float32 w/o type:
        let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(await result.data(), 0, 3.5);
        expectArrayInMeanStdRange(await result.data(), 0, 3.5, EPSILON);
        result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(await result.data(), 0, 4.5);
        expectArrayInMeanStdRange(await result.data(), 0, 4.5, EPSILON);
    });
    it('should return a 4D int32 array', async () => {
        const shape = [5, 5, 5, 5];
        const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(await result.data(), 0, 5);
        expectArrayInMeanStdRange(await result.data(), 0, 5, EPSILON);
    });
    it('should throw error when shape is not integer', () => {
        expect(() => tf.truncatedNormal([5.55, 3.33, 2.22], 0, 5, 'int32', SEED))
            .toThrow();
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJ1bmNhdGVkX25vcm1hbF90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvdHJ1bmNhdGVkX25vcm1hbF90ZXN0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sS0FBSyxFQUFFLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxRQUFRLEVBQUUsaUJBQWlCLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUU1RCxPQUFPLEVBQUMseUJBQXlCLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFdEQsaUJBQWlCLENBQUMsaUJBQWlCLEVBQUUsUUFBUSxFQUFFLEdBQUcsRUFBRTtJQUNsRCx5REFBeUQ7SUFDekQsTUFBTSxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBQ3JCLE1BQU0sSUFBSSxHQUFHLElBQUksQ0FBQztJQUVsQixTQUFTLHFCQUFxQixDQUMxQixNQUFrQixFQUFFLElBQVksRUFBRSxJQUFZO1FBQ2hELE1BQU0sTUFBTSxHQUFHLElBQUksR0FBRyxJQUFJLEdBQUcsQ0FBQyxDQUFDO1FBQy9CLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFO1lBQ3RDLE1BQU0sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsbUJBQW1CLENBQUMsTUFBTSxDQUFDLENBQUM7U0FDekQ7SUFDSCxDQUFDO0lBRUQsRUFBRSxDQUFDLHlDQUF5QyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3ZELE1BQU0sS0FBSyxHQUFhLENBQUMsSUFBSSxDQUFDLENBQUM7UUFFL0IsdUNBQXVDO1FBQ3ZDLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsS0FBSyxFQUFFLENBQUMsRUFBRSxHQUFHLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQzNELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQ3JDLHFCQUFxQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQztRQUNuRCx5QkFBeUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsR0FBRyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBRWhFLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLEtBQUssRUFBRSxDQUFDLEVBQUUsR0FBRyxFQUFFLFNBQVMsRUFBRSxJQUFJLENBQUMsQ0FBQztRQUM1RCxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUNyQyxxQkFBcUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDbkQseUJBQXlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEdBQUcsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUNsRSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyx1Q0FBdUMsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNyRCxNQUFNLEtBQUssR0FBYSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQy9CLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsS0FBSyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsT0FBTyxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQzlELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ25DLHFCQUFxQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNqRCx5QkFBeUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ2hFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGtDQUFrQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ2hELE1BQU0sS0FBSyxHQUFxQixDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQztRQUV6Qyx1Q0FBdUM7UUFDdkMsSUFBSSxNQUFNLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxFQUFFLEdBQUcsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDM0QsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDckMscUJBQXFCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQ25ELHlCQUF5QixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxHQUFHLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFFaEUsTUFBTSxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsS0FBSyxFQUFFLENBQUMsRUFBRSxHQUFHLEVBQUUsU0FBUyxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQzVELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQ3JDLHFCQUFxQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQztRQUNuRCx5QkFBeUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsR0FBRyxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ2xFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGdDQUFnQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzlDLE1BQU0sS0FBSyxHQUFxQixDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQztRQUN6QyxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLEtBQUssRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sRUFBRSxJQUFJLENBQUMsQ0FBQztRQUM5RCxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUNuQyxxQkFBcUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDakQseUJBQXlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUNoRSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxrQ0FBa0MsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNoRCxNQUFNLEtBQUssR0FBNkIsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO1FBRXJELHVDQUF1QztRQUN2QyxJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLEtBQUssRUFBRSxDQUFDLEVBQUUsR0FBRyxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztRQUMzRCxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUNyQyxxQkFBcUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDbkQseUJBQXlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEdBQUcsRUFBRSxPQUFPLENBQUMsQ0FBQztRQUVoRSxNQUFNLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxFQUFFLEdBQUcsRUFBRSxTQUFTLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDNUQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDckMscUJBQXFCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQ25ELHlCQUF5QixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxHQUFHLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDbEUsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsZ0NBQWdDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDOUMsTUFBTSxLQUFLLEdBQTZCLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQztRQUNyRCxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsZUFBZSxDQUFDLEtBQUssRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sRUFBRSxJQUFJLENBQUMsQ0FBQztRQUM5RCxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUNuQyxxQkFBcUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDakQseUJBQXlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUNoRSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxrQ0FBa0MsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNoRCxNQUFNLEtBQUssR0FBcUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUU3RCx1Q0FBdUM7UUFDdkMsSUFBSSxNQUFNLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxFQUFFLEdBQUcsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDM0QsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDckMscUJBQXFCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQ25ELHlCQUF5QixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxHQUFHLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFFaEUsTUFBTSxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsS0FBSyxFQUFFLENBQUMsRUFBRSxHQUFHLEVBQUUsU0FBUyxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQzVELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQ3JDLHFCQUFxQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQztRQUNuRCx5QkFBeUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsR0FBRyxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ2xFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGdDQUFnQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzlDLE1BQU0sS0FBSyxHQUFxQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQzdELE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxlQUFlLENBQUMsS0FBSyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsT0FBTyxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQzlELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ25DLHFCQUFxQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNqRCx5QkFBeUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ2hFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDhDQUE4QyxFQUFFLEdBQUcsRUFBRTtRQUN0RCxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLGVBQWUsQ0FBQyxDQUFDLElBQUksRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLEVBQUUsSUFBSSxDQUFDLENBQUM7YUFDcEUsT0FBTyxFQUFFLENBQUM7SUFDakIsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICogYXMgdGYgZnJvbSAnLi4vaW5kZXgnO1xuaW1wb3J0IHtBTExfRU5WUywgZGVzY3JpYmVXaXRoRmxhZ3N9IGZyb20gJy4uL2phc21pbmVfdXRpbCc7XG5pbXBvcnQge1R5cGVkQXJyYXl9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7ZXhwZWN0QXJyYXlJbk1lYW5TdGRSYW5nZX0gZnJvbSAnLi9yYW5kX3V0aWwnO1xuXG5kZXNjcmliZVdpdGhGbGFncygndHJ1bmNhdGVkTm9ybWFsJywgQUxMX0VOVlMsICgpID0+IHtcbiAgLy8gRXhwZWN0IHNsaWdodGx5IGhpZ2hlciB2YXJpYW5jZXMgZm9yIHRydW5jYXRlZCB2YWx1ZXMuXG4gIGNvbnN0IEVQU0lMT04gPSAwLjYwO1xuICBjb25zdCBTRUVEID0gMjAwMjtcblxuICBmdW5jdGlvbiBhc3NlcnRUcnVuY2F0ZWRWYWx1ZXMoXG4gICAgICB2YWx1ZXM6IFR5cGVkQXJyYXksIG1lYW46IG51bWJlciwgc3RkdjogbnVtYmVyKSB7XG4gICAgY29uc3QgYm91bmRzID0gbWVhbiArIHN0ZHYgKiAyO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBleHBlY3QoTWF0aC5hYnModmFsdWVzW2ldKSkudG9CZUxlc3NUaGFuT3JFcXVhbChib3VuZHMpO1xuICAgIH1cbiAgfVxuXG4gIGl0KCdzaG91bGQgcmV0dXJuIGEgcmFuZG9tIDFEIGZsb2F0MzIgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc2hhcGU6IFtudW1iZXJdID0gWzEwMDBdO1xuXG4gICAgLy8gRW5zdXJlIGRlZmF1bHRzIHRvIGZsb2F0MzIgdy9vIHR5cGU6XG4gICAgbGV0IHJlc3VsdCA9IHRmLnRydW5jYXRlZE5vcm1hbChzaGFwZSwgMCwgMy41LCBudWxsLCBTRUVEKTtcbiAgICBleHBlY3QocmVzdWx0LmR0eXBlKS50b0JlKCdmbG9hdDMyJyk7XG4gICAgYXNzZXJ0VHJ1bmNhdGVkVmFsdWVzKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDMuNSk7XG4gICAgZXhwZWN0QXJyYXlJbk1lYW5TdGRSYW5nZShhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCAzLjUsIEVQU0lMT04pO1xuXG4gICAgcmVzdWx0ID0gdGYudHJ1bmNhdGVkTm9ybWFsKHNoYXBlLCAwLCA0LjUsICdmbG9hdDMyJywgU0VFRCk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnZmxvYXQzMicpO1xuICAgIGFzc2VydFRydW5jYXRlZFZhbHVlcyhhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCA0LjUpO1xuICAgIGV4cGVjdEFycmF5SW5NZWFuU3RkUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgNC41LCBFUFNJTE9OKTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCByZXR1cm4gYSByYW5kb24gMUQgaW50MzIgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc2hhcGU6IFtudW1iZXJdID0gWzEwMDBdO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLnRydW5jYXRlZE5vcm1hbChzaGFwZSwgMCwgNSwgJ2ludDMyJywgU0VFRCk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnaW50MzInKTtcbiAgICBhc3NlcnRUcnVuY2F0ZWRWYWx1ZXMoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgNSk7XG4gICAgZXhwZWN0QXJyYXlJbk1lYW5TdGRSYW5nZShhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCA1LCBFUFNJTE9OKTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCByZXR1cm4gYSAyRCBmbG9hdDMyIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdID0gWzUwLCA1MF07XG5cbiAgICAvLyBFbnN1cmUgZGVmYXVsdHMgdG8gZmxvYXQzMiB3L28gdHlwZTpcbiAgICBsZXQgcmVzdWx0ID0gdGYudHJ1bmNhdGVkTm9ybWFsKHNoYXBlLCAwLCAzLjUsIG51bGwsIFNFRUQpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2Zsb2F0MzInKTtcbiAgICBhc3NlcnRUcnVuY2F0ZWRWYWx1ZXMoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgMy41KTtcbiAgICBleHBlY3RBcnJheUluTWVhblN0ZFJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDMuNSwgRVBTSUxPTik7XG5cbiAgICByZXN1bHQgPSB0Zi50cnVuY2F0ZWROb3JtYWwoc2hhcGUsIDAsIDQuNSwgJ2Zsb2F0MzInLCBTRUVEKTtcbiAgICBleHBlY3QocmVzdWx0LmR0eXBlKS50b0JlKCdmbG9hdDMyJyk7XG4gICAgYXNzZXJ0VHJ1bmNhdGVkVmFsdWVzKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDQuNSk7XG4gICAgZXhwZWN0QXJyYXlJbk1lYW5TdGRSYW5nZShhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCA0LjUsIEVQU0lMT04pO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHJldHVybiBhIDJEIGludDMyIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdID0gWzUwLCA1MF07XG4gICAgY29uc3QgcmVzdWx0ID0gdGYudHJ1bmNhdGVkTm9ybWFsKHNoYXBlLCAwLCA1LCAnaW50MzInLCBTRUVEKTtcbiAgICBleHBlY3QocmVzdWx0LmR0eXBlKS50b0JlKCdpbnQzMicpO1xuICAgIGFzc2VydFRydW5jYXRlZFZhbHVlcyhhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCA1KTtcbiAgICBleHBlY3RBcnJheUluTWVhblN0ZFJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDUsIEVQU0lMT04pO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHJldHVybiBhIDNEIGZsb2F0MzIgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsxMCwgMTAsIDEwXTtcblxuICAgIC8vIEVuc3VyZSBkZWZhdWx0cyB0byBmbG9hdDMyIHcvbyB0eXBlOlxuICAgIGxldCByZXN1bHQgPSB0Zi50cnVuY2F0ZWROb3JtYWwoc2hhcGUsIDAsIDMuNSwgbnVsbCwgU0VFRCk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnZmxvYXQzMicpO1xuICAgIGFzc2VydFRydW5jYXRlZFZhbHVlcyhhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCAzLjUpO1xuICAgIGV4cGVjdEFycmF5SW5NZWFuU3RkUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgMy41LCBFUFNJTE9OKTtcblxuICAgIHJlc3VsdCA9IHRmLnRydW5jYXRlZE5vcm1hbChzaGFwZSwgMCwgNC41LCAnZmxvYXQzMicsIFNFRUQpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2Zsb2F0MzInKTtcbiAgICBhc3NlcnRUcnVuY2F0ZWRWYWx1ZXMoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgNC41KTtcbiAgICBleHBlY3RBcnJheUluTWVhblN0ZFJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDQuNSwgRVBTSUxPTik7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgcmV0dXJuIGEgM0QgaW50MzIgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsxMCwgMTAsIDEwXTtcbiAgICBjb25zdCByZXN1bHQgPSB0Zi50cnVuY2F0ZWROb3JtYWwoc2hhcGUsIDAsIDUsICdpbnQzMicsIFNFRUQpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2ludDMyJyk7XG4gICAgYXNzZXJ0VHJ1bmNhdGVkVmFsdWVzKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDUpO1xuICAgIGV4cGVjdEFycmF5SW5NZWFuU3RkUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgNSwgRVBTSUxPTik7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgcmV0dXJuIGEgNEQgZmxvYXQzMiBhcnJheScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNSwgNSwgNSwgNV07XG5cbiAgICAvLyBFbnN1cmUgZGVmYXVsdHMgdG8gZmxvYXQzMiB3L28gdHlwZTpcbiAgICBsZXQgcmVzdWx0ID0gdGYudHJ1bmNhdGVkTm9ybWFsKHNoYXBlLCAwLCAzLjUsIG51bGwsIFNFRUQpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2Zsb2F0MzInKTtcbiAgICBhc3NlcnRUcnVuY2F0ZWRWYWx1ZXMoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgMy41KTtcbiAgICBleHBlY3RBcnJheUluTWVhblN0ZFJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDMuNSwgRVBTSUxPTik7XG5cbiAgICByZXN1bHQgPSB0Zi50cnVuY2F0ZWROb3JtYWwoc2hhcGUsIDAsIDQuNSwgJ2Zsb2F0MzInLCBTRUVEKTtcbiAgICBleHBlY3QocmVzdWx0LmR0eXBlKS50b0JlKCdmbG9hdDMyJyk7XG4gICAgYXNzZXJ0VHJ1bmNhdGVkVmFsdWVzKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDQuNSk7XG4gICAgZXhwZWN0QXJyYXlJbk1lYW5TdGRSYW5nZShhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCA0LjUsIEVQU0lMT04pO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHJldHVybiBhIDREIGludDMyIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs1LCA1LCA1LCA1XTtcbiAgICBjb25zdCByZXN1bHQgPSB0Zi50cnVuY2F0ZWROb3JtYWwoc2hhcGUsIDAsIDUsICdpbnQzMicsIFNFRUQpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2ludDMyJyk7XG4gICAgYXNzZXJ0VHJ1bmNhdGVkVmFsdWVzKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDUpO1xuICAgIGV4cGVjdEFycmF5SW5NZWFuU3RkUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgNSwgRVBTSUxPTik7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgdGhyb3cgZXJyb3Igd2hlbiBzaGFwZSBpcyBub3QgaW50ZWdlcicsICgpID0+IHtcbiAgICBleHBlY3QoKCkgPT4gdGYudHJ1bmNhdGVkTm9ybWFsKFs1LjU1LCAzLjMzLCAyLjIyXSwgMCwgNSwgJ2ludDMyJywgU0VFRCkpXG4gICAgICAgIC50b1Rocm93KCk7XG4gIH0pO1xufSk7XG4iXX0=