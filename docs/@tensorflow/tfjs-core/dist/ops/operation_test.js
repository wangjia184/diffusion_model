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
import { ALL_ENVS, describeWithFlags } from '../jasmine_util';
import { op } from './operation';
describeWithFlags('operation', ALL_ENVS, () => {
    it('executes and preserves function name', () => {
        const f = () => 2;
        const opfn = /* @__PURE__ */ op({ 'opName': f });
        expect(opfn.name).toBe('opName__op');
        expect(opfn()).toBe(2);
    });
    it('executes, preserves function name, strips underscore', () => {
        const f = () => 2;
        const opfn = /* @__PURE__ */ op({ 'opName_': f });
        expect(opfn.name).toBe('opName__op');
        expect(opfn()).toBe(2);
    });
    it('throws when passing an object with multiple keys', () => {
        const f = () => 2;
        expect(() => op({ 'opName_': f, 'opName2_': f }))
            .toThrowError(/Please provide an object with a single key/);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoib3BlcmF0aW9uX3Rlc3QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9vcGVyYXRpb25fdGVzdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLEVBQUMsUUFBUSxFQUFFLGlCQUFpQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFDNUQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQixpQkFBaUIsQ0FBQyxXQUFXLEVBQUUsUUFBUSxFQUFFLEdBQUcsRUFBRTtJQUM1QyxFQUFFLENBQUMsc0NBQXNDLEVBQUUsR0FBRyxFQUFFO1FBQzlDLE1BQU0sQ0FBQyxHQUFHLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNsQixNQUFNLElBQUksR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsUUFBUSxFQUFFLENBQUMsRUFBQyxDQUFDLENBQUM7UUFFL0MsTUFBTSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUM7UUFDckMsTUFBTSxDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pCLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNEQUFzRCxFQUFFLEdBQUcsRUFBRTtRQUM5RCxNQUFNLENBQUMsR0FBRyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDbEIsTUFBTSxJQUFJLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFNBQVMsRUFBRSxDQUFDLEVBQUMsQ0FBQyxDQUFDO1FBRWhELE1BQU0sQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLFlBQVksQ0FBQyxDQUFDO1FBQ3JDLE1BQU0sQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN6QixDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxrREFBa0QsRUFBRSxHQUFHLEVBQUU7UUFDMUQsTUFBTSxDQUFDLEdBQUcsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ2xCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBQyxTQUFTLEVBQUUsQ0FBQyxFQUFFLFVBQVUsRUFBRSxDQUFDLEVBQUMsQ0FBQyxDQUFDO2FBQzFDLFlBQVksQ0FBQyw0Q0FBNEMsQ0FBQyxDQUFDO0lBQ2xFLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge0FMTF9FTlZTLCBkZXNjcmliZVdpdGhGbGFnc30gZnJvbSAnLi4vamFzbWluZV91dGlsJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ29wZXJhdGlvbicsIEFMTF9FTlZTLCAoKSA9PiB7XG4gIGl0KCdleGVjdXRlcyBhbmQgcHJlc2VydmVzIGZ1bmN0aW9uIG5hbWUnLCAoKSA9PiB7XG4gICAgY29uc3QgZiA9ICgpID0+IDI7XG4gICAgY29uc3Qgb3BmbiA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7J29wTmFtZSc6IGZ9KTtcblxuICAgIGV4cGVjdChvcGZuLm5hbWUpLnRvQmUoJ29wTmFtZV9fb3AnKTtcbiAgICBleHBlY3Qob3BmbigpKS50b0JlKDIpO1xuICB9KTtcblxuICBpdCgnZXhlY3V0ZXMsIHByZXNlcnZlcyBmdW5jdGlvbiBuYW1lLCBzdHJpcHMgdW5kZXJzY29yZScsICgpID0+IHtcbiAgICBjb25zdCBmID0gKCkgPT4gMjtcbiAgICBjb25zdCBvcGZuID0gLyogQF9fUFVSRV9fICovIG9wKHsnb3BOYW1lXyc6IGZ9KTtcblxuICAgIGV4cGVjdChvcGZuLm5hbWUpLnRvQmUoJ29wTmFtZV9fb3AnKTtcbiAgICBleHBlY3Qob3BmbigpKS50b0JlKDIpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIHdoZW4gcGFzc2luZyBhbiBvYmplY3Qgd2l0aCBtdWx0aXBsZSBrZXlzJywgKCkgPT4ge1xuICAgIGNvbnN0IGYgPSAoKSA9PiAyO1xuICAgIGV4cGVjdCgoKSA9PiBvcCh7J29wTmFtZV8nOiBmLCAnb3BOYW1lMl8nOiBmfSkpXG4gICAgICAgIC50b1Rocm93RXJyb3IoL1BsZWFzZSBwcm92aWRlIGFuIG9iamVjdCB3aXRoIGEgc2luZ2xlIGtleS8pO1xuICB9KTtcbn0pO1xuIl19