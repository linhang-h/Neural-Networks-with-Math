# FranklinTheorems package for amsthms
# Includes the Package, bringing the lx_functions into scope
# using FranklinTheorems
# Includes the custom markdown files, bringing the `\newcommand` and `\newenvironment` definitions into scope.
# Franklin.include_external_config(FranklinTheorems.config_path()) 


function hfun_bar(vname)
    val = Meta.parse(vname[1])
    return round(sqrt(val), digits=2)
end

function hfun_m1fill(vname)
    var = vname[1]
    return pagevar("index", var)
end

function lx_baz(com, _)
    # keep this first line
    brace_content = Franklin.content(com.braces[1]) # input string
    # do whatever you want here
    return uppercase(brace_content)
end

function hfun_custom_taglist()::String
    tag = locvar(:fd_tag)
    rpaths = globvar("fd_tag_pages")[tag]
    sorter(p) = begin
        pubdate = pagevar(p, :published)
        if isnothing(pubdate)
            return Date(Dates.unix2datetime(stat(p * ".md").ctime))
        end
        return Date(pubdate, dateformat"d U Y")
    end
    sort!(rpaths, by=sorter, rev=true)

    io = IOBuffer()
    write(io, """<ul class="blog-posts">""")
    # go over all paths
    for rpath in rpaths
        write(io, "<li><span><i>")
        url = get_url(rpath)
        title = pagevar(rpath, :title)
        pubdate = pagevar(rpath, :published)
        if isnothing(pubdate)
            date    = "$curyear-$curmonth-$curday"
        else
            date    = Date(pubdate, dateformat"d U Y")
        end
        # write some appropriate HTML
        write(io, """$date</i></span><a href="$url">$title</a>""")
    end
    write(io, "</ul>")
    return String(take!(io))
end

"""
    {{blogposts}}

Plug in the list of blog posts contained in the `/posts` folder.
Souce: <https://github.com/abhishalya/abhishalya.github.io>.
"""
@delay function hfun_blogposts()
    list = readdir("posts")

    filter!(endswith(".md"), list)
    function sorter(p)
        ps = splitext(p)[1]
        url = "posts/$ps/"
        surl = strip(url, '/')
        pubdate = pagevar(surl, "rss_pubdate")
    end
    sort!(list, by=sorter, rev=true)

    io = IOBuffer()

    write(io, """<div class="franklin-content">""")
    write(io, """<div class="blog-posts">""")
    for (i, post) in enumerate(list)
        if post == "index.md"
            continue
        end
        ps = splitext(post)[1]
        url = "posts/$ps/"
        url_aux = "../posts/$ps/"
        surl = strip(url, '/')
        title = pagevar(surl, "title")
        excerpt = pagevar(surl, "excerpt")
        image = pagevar(surl, "image")
        authors = pagevar(surl, "authors")
        write(io, """
        <div class="blog-card">
            <img src=$image alt=$title class="blog-image">
            <div class="blog-content">
                <h4 class="blog-title">$title</h3>
                <h4 class="blog-author">$authors</h4>
                <p class="blog-excerpt">$excerpt</p>
                <a href=$url_aux class="read-more">Read More →</a>
            </div>
        </div>
        """)
    end
    write(io, "</div>")
    write(io, "</div>")
    return String(take!(io))
end