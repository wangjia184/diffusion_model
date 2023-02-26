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
 *
 * =============================================================================
 */
import { RingBuffer } from './ring_buffer';
export class GrowingRingBuffer extends RingBuffer {
    /**
     * Constructs a `GrowingRingBuffer`.
     */
    constructor() {
        super(GrowingRingBuffer.INITIAL_CAPACITY);
    }
    isFull() {
        return false;
    }
    push(value) {
        if (super.isFull()) {
            this.expand();
        }
        super.push(value);
    }
    unshift(value) {
        if (super.isFull()) {
            this.expand();
        }
        super.unshift(value);
    }
    /**
     * Doubles the capacity of the buffer.
     */
    expand() {
        const newCapacity = this.capacity * 2;
        const newData = new Array(newCapacity);
        const len = this.length();
        // Rotate the buffer to start at index 0 again, since we can't just
        // allocate more space at the end.
        for (let i = 0; i < len; i++) {
            newData[i] = this.get(this.wrap(this.begin + i));
        }
        this.data = newData;
        this.capacity = newCapacity;
        this.doubledCapacity = 2 * this.capacity;
        this.begin = 0;
        this.end = len;
    }
}
GrowingRingBuffer.INITIAL_CAPACITY = 32;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3Jvd2luZ19yaW5nX2J1ZmZlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtZGF0YS9zcmMvdXRpbC9ncm93aW5nX3JpbmdfYnVmZmVyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7O0dBZ0JHO0FBRUgsT0FBTyxFQUFDLFVBQVUsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUV6QyxNQUFNLE9BQU8saUJBQXFCLFNBQVEsVUFBYTtJQUdyRDs7T0FFRztJQUNIO1FBQ0UsS0FBSyxDQUFDLGlCQUFpQixDQUFDLGdCQUFnQixDQUFDLENBQUM7SUFDNUMsQ0FBQztJQUVRLE1BQU07UUFDYixPQUFPLEtBQUssQ0FBQztJQUNmLENBQUM7SUFFUSxJQUFJLENBQUMsS0FBUTtRQUNwQixJQUFJLEtBQUssQ0FBQyxNQUFNLEVBQUUsRUFBRTtZQUNsQixJQUFJLENBQUMsTUFBTSxFQUFFLENBQUM7U0FDZjtRQUNELEtBQUssQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDcEIsQ0FBQztJQUVRLE9BQU8sQ0FBQyxLQUFRO1FBQ3ZCLElBQUksS0FBSyxDQUFDLE1BQU0sRUFBRSxFQUFFO1lBQ2xCLElBQUksQ0FBQyxNQUFNLEVBQUUsQ0FBQztTQUNmO1FBQ0QsS0FBSyxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUN2QixDQUFDO0lBRUQ7O09BRUc7SUFDSyxNQUFNO1FBQ1osTUFBTSxXQUFXLEdBQUcsSUFBSSxDQUFDLFFBQVEsR0FBRyxDQUFDLENBQUM7UUFDdEMsTUFBTSxPQUFPLEdBQUcsSUFBSSxLQUFLLENBQUksV0FBVyxDQUFDLENBQUM7UUFDMUMsTUFBTSxHQUFHLEdBQUcsSUFBSSxDQUFDLE1BQU0sRUFBRSxDQUFDO1FBRTFCLG1FQUFtRTtRQUNuRSxrQ0FBa0M7UUFDbEMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEdBQUcsRUFBRSxDQUFDLEVBQUUsRUFBRTtZQUM1QixPQUFPLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUNsRDtRQUVELElBQUksQ0FBQyxJQUFJLEdBQUcsT0FBTyxDQUFDO1FBQ3BCLElBQUksQ0FBQyxRQUFRLEdBQUcsV0FBVyxDQUFDO1FBQzVCLElBQUksQ0FBQyxlQUFlLEdBQUcsQ0FBQyxHQUFHLElBQUksQ0FBQyxRQUFRLENBQUM7UUFDekMsSUFBSSxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUM7UUFDZixJQUFJLENBQUMsR0FBRyxHQUFHLEdBQUcsQ0FBQztJQUNqQixDQUFDOztBQTlDYyxrQ0FBZ0IsR0FBRyxFQUFFLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1JpbmdCdWZmZXJ9IGZyb20gJy4vcmluZ19idWZmZXInO1xuXG5leHBvcnQgY2xhc3MgR3Jvd2luZ1JpbmdCdWZmZXI8VD4gZXh0ZW5kcyBSaW5nQnVmZmVyPFQ+IHtcbiAgcHJpdmF0ZSBzdGF0aWMgSU5JVElBTF9DQVBBQ0lUWSA9IDMyO1xuXG4gIC8qKlxuICAgKiBDb25zdHJ1Y3RzIGEgYEdyb3dpbmdSaW5nQnVmZmVyYC5cbiAgICovXG4gIGNvbnN0cnVjdG9yKCkge1xuICAgIHN1cGVyKEdyb3dpbmdSaW5nQnVmZmVyLklOSVRJQUxfQ0FQQUNJVFkpO1xuICB9XG5cbiAgb3ZlcnJpZGUgaXNGdWxsKCkge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIG92ZXJyaWRlIHB1c2godmFsdWU6IFQpIHtcbiAgICBpZiAoc3VwZXIuaXNGdWxsKCkpIHtcbiAgICAgIHRoaXMuZXhwYW5kKCk7XG4gICAgfVxuICAgIHN1cGVyLnB1c2godmFsdWUpO1xuICB9XG5cbiAgb3ZlcnJpZGUgdW5zaGlmdCh2YWx1ZTogVCkge1xuICAgIGlmIChzdXBlci5pc0Z1bGwoKSkge1xuICAgICAgdGhpcy5leHBhbmQoKTtcbiAgICB9XG4gICAgc3VwZXIudW5zaGlmdCh2YWx1ZSk7XG4gIH1cblxuICAvKipcbiAgICogRG91YmxlcyB0aGUgY2FwYWNpdHkgb2YgdGhlIGJ1ZmZlci5cbiAgICovXG4gIHByaXZhdGUgZXhwYW5kKCkge1xuICAgIGNvbnN0IG5ld0NhcGFjaXR5ID0gdGhpcy5jYXBhY2l0eSAqIDI7XG4gICAgY29uc3QgbmV3RGF0YSA9IG5ldyBBcnJheTxUPihuZXdDYXBhY2l0eSk7XG4gICAgY29uc3QgbGVuID0gdGhpcy5sZW5ndGgoKTtcblxuICAgIC8vIFJvdGF0ZSB0aGUgYnVmZmVyIHRvIHN0YXJ0IGF0IGluZGV4IDAgYWdhaW4sIHNpbmNlIHdlIGNhbid0IGp1c3RcbiAgICAvLyBhbGxvY2F0ZSBtb3JlIHNwYWNlIGF0IHRoZSBlbmQuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsZW47IGkrKykge1xuICAgICAgbmV3RGF0YVtpXSA9IHRoaXMuZ2V0KHRoaXMud3JhcCh0aGlzLmJlZ2luICsgaSkpO1xuICAgIH1cblxuICAgIHRoaXMuZGF0YSA9IG5ld0RhdGE7XG4gICAgdGhpcy5jYXBhY2l0eSA9IG5ld0NhcGFjaXR5O1xuICAgIHRoaXMuZG91YmxlZENhcGFjaXR5ID0gMiAqIHRoaXMuY2FwYWNpdHk7XG4gICAgdGhpcy5iZWdpbiA9IDA7XG4gICAgdGhpcy5lbmQgPSBsZW47XG4gIH1cbn1cbiJdfQ==