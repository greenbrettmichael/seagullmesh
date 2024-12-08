template<typename T>
class Indices {
    typedef typename T::size_type size_type;
    typedef typename py::array_t<size_type> A;

    private:
        // reference? smart pointer? unique pointer?
        // Should be able to slice an Indices and get a Indices<view>
        Array idxs;

    public:
        Indices() :  {}
        Indices(Array idxs) : idxs(idxs) {}

        static Indices from_vector(const std::vector<T>& descriptors) {
            // Convert the descriptors into raw size_type
            Array idxs({descriptors.size()});
            auto r = idxs.mutable_unchecked<1>();
            for (size_t i = 0; i < n; ++i) {
                r(i) = size_type(descriptors[i]);
            }
            return Indices{idxs};
        }

        // Allow arbitary array sizes and shapes as long as we can iterate over 1 d
        size_t size() {
            return idxs.num_elements();
        }

        // Placeholder until we can write a range adaptor
        std::vector<T> to_vector() {
            std::vector<T> out;
            out.reserve(n);
            auto r = idxs.unchecked<1>();
            for (size_t i = 0; i < n; ++i) {
                out.emplace_back(T(r(i)));
            }
            return out;
        }
}